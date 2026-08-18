#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import torch
import json
import numpy as np
from random import randint
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, SaveImg
import uuid
from tqdm import tqdm
from utils.image_utils import *
from utils.camera_utils import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DOWNSAMPLE = 2
alpha = 0.1
beta = 0.1

def training(dataset, opt, pipe, testing_iterations, pruning_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, use_mask=args.use_mask, use_spin=args.use_spin)
    bars_path = os.path.join(args.source_path, "ref_image/data.json")
    if args.use_mask and os.path.exists(bars_path):
        print(f"{bars_path} exist, using preset bars.")
        with open(bars_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        setattr(scene, "bars", data)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # bg_color = [0.13725490196078433, 0.29411764705882354, 0.8627450980392157]
    print(f"background color is {bg_color}")
    print(f"Training scenes: {len(scene.getTrainCameras())}, testing scenes: {len(scene.getTestCameras())}")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    global alpha, beta
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()# + scene.getTestCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if args.use_mask: 
            mask = viewpoint_cam.mask.cuda()
            # bar = 0.9 if not hasattr(scene, "bars") else scene.bars[viewpoint_cam.image_name]
            # mask = (mask > bar).float()
            # mask = erode_mask(mask, iterations=5)
            # mask = mask + (1 - mask) * beta
        else:
            mask = None
        
        Ll1 = l1_loss(image, gt_image, mask=mask)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))

        if viewpoint_cam.ref_image is not None and alpha > 1e-5 and args.use_mask:
            ref_mask = (1 - mask)
            ref_gt_image = viewpoint_cam.ref_image.cuda()
            ref_mask = ref_mask + (1 - ref_mask)
            ref_Ll1 = l1_loss(image, ref_gt_image, mask=ref_mask)
            ref_loss = (1.0 - opt.lambda_dssim) * ref_Ll1 + opt.lambda_dssim * (1.0 - ssim(image, ref_gt_image, mask=ref_mask))
            loss = loss + alpha * ref_loss
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            vis_count = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, pruning_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                

            # Densification
            if iteration < opt.densify_until_iter and iteration not in pruning_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif iteration in pruning_iterations:
                gaussians.prune_float_points(viewpoint_stack, vis_count)
            if iteration > opt.densify_until_iter:
                alpha = 0.1
                beta = 0.1

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, pruning_iterations, scene : Scene, renderFunc, renderArgs):
    vis_count = torch.zeros(scene.gaussians._xyz.shape[0]).to(scene.gaussians._xyz.device)
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        image_name2points = {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    renderpkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderpkg["render"], 0.0, 1.0)
                    depth = renderpkg["depth"]
                    vis_count = vis_count + renderpkg["visibility_filter"]
                    depth = 1 / depth / 100
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    points = depth_image_to_point_cloud(
                        image.detach().cpu(), 
                        depth.detach().cpu(), 
                        viewpoint.FoVx, 
                        viewpoint.FoVy, 
                        viewpoint.world_view_transform.cpu(), 
                        camera_center=viewpoint.camera_center.cpu(),
                        downsample=DOWNSAMPLE
                        )
                    image_name2points[viewpoint.image_name] = points
                    if tb_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    SaveImg(image, os.path.join(tb_writer.log_dir, f"view_{iteration}_render"), viewpoint.image_name + ".jpg")
                    # SaveImg(depth, os.path.join(tb_writer.log_dir, f"view_{iteration}_depth"), viewpoint.image_name + ".jpg")
                    # write_point_cloud(os.path.join(tb_writer.log_dir, f"view_{iteration}_render/", viewpoint.image_name + "_depth.ply"), points)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        camera_idx = 0
        
        for map, p in K_neighbor_camera(config['cameras'], image_name2points, args, downsample=DOWNSAMPLE):
            SaveImg(torch.from_numpy(map), os.path.join(tb_writer.log_dir, f"view_{iteration}_confidence"), config['cameras'][camera_idx].image_name + ".jpg")
            camera_idx += 1
            
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    elif iteration in pruning_iterations:
        for idx, viewpoint in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
            renderpkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            # print(depth)
            # depth = (depth - depth.min()) / (depth.max() - depth.min())
            vis_count = vis_count + renderpkg["visibility_filter"]
        
    return vis_count

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--pruning_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_spin", action="store_true")
    parser.add_argument("--use_mask", action="store_true", help="启用 mask 功能")
    parser.add_argument("--no_use_mask", dest="use_mask", action="store_false", help="禁用 mask 功能")
    parser.set_defaults(use_mask=True)  # 默认值

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # if os.path.exists(os.path.join(args.model_path, f"view_{args.iterations}_render")):
    #     print(f"Skip training {args.model_path}")
    #     exit(0)
    print("Optimizing " + args.model_path + f" with {'' if args.use_mask else 'no '}mask")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.pruning_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
