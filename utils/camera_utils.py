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

from scene.cameras import Camera
import numpy as np
import math
import torch
import random
from scipy import stats
from scipy.spatial import KDTree
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage import img_as_float
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from utils.image_utils import rgb_to_lab, ciede2000_pairwise

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    mask = ref_image = ref_mask = depth = None
    if cam_info.mask is not None:
        mask = cam_info.mask.convert('L')
        mask = PILtoTorch(mask, resolution)
        
    if cam_info.ref_image is not None:
        ref_image = PILtoTorch(cam_info.ref_image, resolution)
    
    if cam_info.ref_mask is not None:
        ref_mask = cam_info.ref_mask.convert('L')
        ref_mask = PILtoTorch(ref_mask, resolution)
        

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, mask=mask, ref_image=ref_image, ref_mask=ref_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_intrinsic_transform(fov_x, fov_y, pixel_width, pixel_height):
    camera_intrinsics = np.zeros((3,4))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(fov_x/2.0)
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(fov_y/2.0)
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics


def depth_image_to_point_cloud(
    rgb: torch.Tensor, 
    depth: torch.Tensor, 
    fov_x, fov_y, pose,
    camera_center,
    scale=100,
    downsample=2,
    ):
    rgb, depth, pose, camera_center = (rgb.clamp(0, 1.0) * 255).permute(1, 2, 0).numpy(), depth.squeeze(0).numpy(), pose.numpy(), camera_center.numpy()
    # check_C2W(pose, camera_center)
    C2W = np.linalg.inv(pose).T
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    K = camera_intrinsic_transform(fov_x, fov_y, rgb.shape[1], rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    depth[np.where(depth >= 1.0)] = 1.0 
    Z = depth.astype(float) * scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    
    X, Y, Z = X[::downsample, ::downsample], Y[::downsample, ::downsample], Z[::downsample, ::downsample]
    rgb = rgb[::downsample, ::downsample, :]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = C2W @ position

    B = np.ravel(rgb[:, :, 0]).clip(0, 255)# [valid]
    G = np.ravel(rgb[:, :, 1]).clip(0, 255)# [valid]
    R = np.ravel(rgb[:, :, 2]).clip(0, 255)# [valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B)))

    return points

def write_point_cloud(ply_filename, points):
    points = points.tolist()
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()

def cosine_sim(ref_color, target_color):
    similarity = torch.cosine_similarity(
            torch.from_numpy(ref_color), 
            torch.from_numpy(target_color)
            ).numpy()
    similarity = (similarity + 1.0) / 2.0
    return similarity

def cie2000_sim(ref_lab, target_lab):
    ref_lab, target_lab = ref_lab.astype(np.int32), target_lab.astype(np.int32)
    ref_lab[:,0], target_lab[:,0] = ref_lab[:,0].clip(0, 100), target_lab[:,0].clip(0, 100)
    ref_lab[:,1], target_lab[:,1] = ref_lab[:,1].clip(-128, 127), target_lab[:,1].clip(-128, 127)
    ref_lab[:,2], target_lab[:,2] = ref_lab[:,2].clip(-128, 127), target_lab[:,2].clip(-128, 127)
    # for i in range(len(ref_lab)):
    #     print(ref_lab[i], target_lab[i])
    #     diff = deltaE_ciede2000(ref_lab[i].astype(np.float64), target_lab[i].astype(np.float64))
    diff = deltaE_ciede2000(ref_lab.astype(np.float64), target_lab.astype(np.float64))
    diff[np.isnan(diff)] = 0.0
    diff = (100.0 - diff) / 100.0
    return diff.clip(0, 1.0)


def cie2000_sim_gpu(ref_lab: torch.Tensor, target_lab: torch.Tensor) -> torch.Tensor:
    """
    ref_lab: Tensor of shape [N, k, 3] or [N, 3]
    target_lab: Tensor of shape [N, k, 3] or [N, 3]
    Returns: similarity scores in [0, 1], shape [N, k] or [N]
    """
    ref_lab = ref_lab.float()
    target_lab = target_lab.float()

    # Clip Lab ranges
    ref_lab[..., 0] = ref_lab[..., 0].clamp(0, 100)
    target_lab[..., 0] = target_lab[..., 0].clamp(0, 100)
    ref_lab[..., 1:] = ref_lab[..., 1:].clamp(-128, 127)
    target_lab[..., 1:] = target_lab[..., 1:].clamp(-128, 127)

    original_shape = ref_lab.shape[:-1]  # [N, k] or [N]

    # Flatten to [M, 3]
    ref_flat = ref_lab.reshape(-1, 3)
    tgt_flat = target_lab.reshape(-1, 3)

    # Compute deltaE
    delta_e = ciede2000_pairwise(ref_flat, tgt_flat)  # [M]
    delta_e = torch.nan_to_num(delta_e, nan=0.0)

    # Convert to similarity
    sim = (100.0 - delta_e) / 100.0
    sim = sim.clamp(0.0, 1.0)

    return sim.reshape(original_shape)

    
@torch.no_grad()    
def points2confidence(
    target_points, 
    ref_points, 
    kd_trees, 
    visible_maps, 
    width, height, 
    dis_bound=float("Inf"),
    sim_fn=cie2000_sim_gpu,
    downsample=2,
    k=8):
    width, height = math.ceil(width / downsample), math.ceil(height / downsample)
    # print(width, height, target_points.shape[0])
    assert width * height == target_points.shape[0]

    avg_map = np.ones((height, width)) * 0.9
    count = np.ones((height, width))
    
    if sim_fn is cie2000_sim:
        target_lab = rgb2lab(target_points[:, 3:] / 255.0)
    elif sim_fn is cie2000_sim_gpu:
        target_lab = rgb_to_lab(
            torch.tensor(target_points[:, 3:] / 255.0).cuda()
            ).unsqueeze(1).expand(-1, k, -1) # N*k*3
    for idx, kd_tree in enumerate(kd_trees):
        distances, point_idxs = kd_tree.query(target_points[:,:3], k=k)
        visible_map = (visible_maps[idx] > 0.5).astype(np.float64)
        similarity = distance = None
        if sim_fn is cie2000_sim:
            for k_idx in range(k):
                dis, point_idx = distances[:, k_idx], point_idxs[:, k_idx]
                
                ref_lab = rgb2lab(ref_points[idx][point_idx, 3:] / 255.0)
                sim = sim_fn(ref_lab, target_lab)
                # else:
                #     sim = sim_fn(ref_points[idx][point_idx, 3:], target_points[:, 3:])
                if similarity is None:
                    similarity = sim
                    distance = dis
                else:
                    mask = (similarity > sim)
                    distance = np.where(mask, distance, dis)
                    similarity =  np.maximum(similarity, sim)
            # reserve the points close to target points
            map = ((distance < dis_bound) * similarity).clip(0, 1.0)
            if visible_maps is not None:
                map = map * visible_map
                visible_map = visible_map * (distance < dis_bound)
            map[np.isnan(map)] = 0.0
            map = map.reshape(height, width)
            avg_map = avg_map + map
            count = count + (np.ones((height, width)) if visible_maps is None else visible_map.reshape(height, width))
        elif sim_fn is cie2000_sim_gpu:
            ref_rgb = torch.stack([
                rgb_to_lab(
                    torch.tensor(ref_points[idx][point_idxs[:, i], 3:] / 255.0).cuda()
                    )  # [N, 3]
                for i in range(k)
            ], dim=1)  # [N, k, 3]
            sim = sim_fn(target_lab, ref_rgb) # [N, k]
            
            dis = torch.tensor(distances[:, :k]).cuda()
            max_sim, max_idx = torch.max(sim, dim=1)  # [N]
            best_dis = dis.gather(1, max_idx.unsqueeze(1)).squeeze(1)  # [N]

            # 输出
            similarity = max_sim
            distance = best_dis
            
            map = (best_dis < dis_bound) * similarity
            if visible_maps is not None:
                visible_map = torch.tensor(visible_map).cuda()
                map = map * visible_map
                visible_map = visible_map * (distance < dis_bound)
            map = torch.nan_to_num(map, nan=0.0)
            map = map.reshape(height, width)
            avg_map = avg_map + map.cpu().numpy()
            if visible_maps is None:
                count = count + np.ones((height, width))
            else:
                count = count + visible_map.cpu().numpy().reshape(height, width)
            
    avg_map = (avg_map / count)# * (count > 1) + (count <= 1) * 0.5
        
    # sample_points = np.stack(sample_points) # K * (h * w) * 3
    # sample_points = sample_points.transpose((1, 0, 2)) # (h * w) * K * 3
    p = None
    # t, p = stats.ttest_1samp(sample_points[:,:,:], target_points[:,:3], axis=1, nan_policy='omit') # (h * w) * 3
    # fill nan with mean
    # p[np.isnan(p)] = 1.0 # np.nanmean(p)
    # transform 3-d p to 1-d
    # p = p.reshape(height, width, 3).mean(axis=2)
    
    return avg_map, p

def get_points_visible_map(points, 
                           cameras, 
                           target_camera,
                           angle_bound=120):
    position = np.vstack((points[:, 0], points[:, 1], points[:, 2], np.ones(len(points[:, 0]))))
    res = []
    for camera in cameras:
        # ndc filter
        ndc_position = np.transpose(np.dot(camera.world_view_transform.cpu().numpy(), position))
        ndc = ndc_position[:,:3] / ndc_position[:,3:]
        
        visible = (ndc <= 1) & (ndc >= -1)
        # ndc positions locate in [-1, 1]
        visible = visible[:, 0] & visible[:, 1] & visible[:, 2]
        visible = visible.astype(np.float64)
        visible[np.isnan(visible)] = 0.0
        
        # angle fliter
        c_1 = camera.camera_center.cpu().numpy().reshape(1, 3)
        c_2 = target_camera.camera_center.cpu().numpy().reshape(1, 3)
        v1 = points[:,:3] - c_1
        v2 = points[:,:3] - c_2
        dots = (v1 * v2).sum(axis=1)
        
        norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        cos_angles = dots / (norms + 1e-9)
        angle_invisible = cos_angles < np.cos(np.deg2rad(angle_bound))
        visible[angle_invisible] = 0.0
        
        res.append(visible)
    return res
            
def K_neighbor_camera(cameras, image_name2points, args, K=12, downsample=2):
    pos = np.stack([camera.camera_center.cpu().numpy() for camera in cameras])
    kd_tree: KDTree = KDTree(data=pos)
    points = [image_name2points[camera.image_name] for camera in cameras]
    points_kd_trees = [KDTree(point[:, 0:3]) for point in points]
    mean_arr = 0.0
    for idx in range(len(cameras)):
        random_num = 0
        distance, point_idx = kd_tree.query([cameras[idx].camera_center.cpu().numpy()], k=list(range(1, K + 2 - random_num)))
        distance, point_idx = distance[0][1:], point_idx[0][1:]
        
        point_idx = random.sample(
            list(
                set(list(range(len(cameras)))) - set([idx] + list(point_idx))
            ),
            random_num
            ) + list(point_idx)
        
        print(f"Generating cofidence map for camera {cameras[idx].image_name} by camera: {[cameras[i].image_name for i in point_idx]}")
        
        ref_trees = [points_kd_trees[i] for i in point_idx]
        ref_points = [points[i] for i in point_idx]
        ref_cameras = [cameras[i] for i in point_idx]
        target_points = points[idx]
        # mask = cameras[idx].mask.cpu().squeeze(0)
        
        # print("using visible_maps...")
        
        visible_maps = get_points_visible_map(target_points, ref_cameras, cameras[idx])
        map, p = points2confidence(
            target_points, 
            ref_points, 
            ref_trees, 
            visible_maps,
            cameras[idx].image_width, 
            cameras[idx].image_height,
            downsample=downsample)
        
        
        map[np.isnan(map)] = 0.0
        mean_arr += map.mean()
        yield map, p
    print(f"Mean vars: {mean_arr / len(cameras)}")
    
