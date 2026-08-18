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

import torch
import torch.nn.functional as Fn

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def rgb_to_lab(rgb):
    """
    Convert RGB tensor [N, 3] in range [0, 1] to Lab tensor [N, 3],
    aligned with skimage.color.rgb2lab.
    """
    # Step 1: sRGB to linear RGB
    threshold = 0.04045
    rgb_linear = torch.where(
        rgb > threshold,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92
    )

    # Step 2: linear RGB to XYZ (D65)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)
    xyz = rgb_linear @ M.T

    # Normalize by reference white point (D65)
    xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], dtype=rgb.dtype, device=rgb.device)
    xyz = xyz / xyz_ref

    # Step 3: XYZ to Lab
    epsilon = 0.008856
    kappa = 903.3

    def f(t):
        return torch.where(t > epsilon, t.pow(1/3), (kappa * t + 16) / 116)

    fx = f(xyz[:, 0])
    fy = f(xyz[:, 1])
    fz = f(xyz[:, 2])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return torch.stack([L, a, b], dim=1)


def ciede2000_pairwise(lab1, lab2):
    """
    lab1: Tensor of shape [N, 3]
    lab2: Tensor of shape [N, 3]
    Returns: Tensor of shape [N, 1] with CIEDE2000 distances
    """
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    avg_L = (L1 + L2) / 2
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2

    G = 0.5 * (1 - torch.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = torch.sqrt(a1p**2 + b1**2)
    C2p = torch.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2

    h1p = torch.atan2(b1, a1p)
    h2p = torch.atan2(b2, a2p)
    h1p = torch.where(h1p < 0, h1p + 2 * torch.pi, h1p)
    h2p = torch.where(h2p < 0, h2p + 2 * torch.pi, h2p)

    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(torch.abs(dhp) > torch.pi, dhp - torch.sign(dhp) * 2 * torch.pi, dhp)
    delta_hp = 2 * torch.sqrt(C1p * C2p) * torch.sin(dhp / 2)

    avg_Hp = h1p + dhp / 2
    avg_Hp = torch.where(torch.abs(h1p - h2p) > torch.pi, avg_Hp + torch.pi, avg_Hp)
    avg_Hp = torch.where(avg_Hp >= 2 * torch.pi, avg_Hp - 2 * torch.pi, avg_Hp)

    T = 1 - 0.17 * torch.cos(avg_Hp - torch.pi / 6) + \
        0.24 * torch.cos(2 * avg_Hp) + \
        0.32 * torch.cos(3 * avg_Hp + torch.pi / 30) - \
        0.20 * torch.cos(4 * avg_Hp - 63 * torch.pi / 180)

    delta_theta = 30 * torch.exp(-((avg_Hp * 180 / torch.pi - 275) / 25)**2)
    Rc = 2 * torch.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7))
    Sl = 1 + (0.015 * (avg_L - 50)**2) / torch.sqrt(20 + (avg_L - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -torch.sin(2 * delta_theta * torch.pi / 180) * Rc

    delta_E = torch.sqrt(
        (delta_Lp / Sl)**2 +
        (delta_Cp / Sc)**2 +
        (delta_hp / Sh)**2 +
        Rt * (delta_Cp / Sc) * (delta_hp / Sh)
    )

    return delta_E.unsqueeze(1)  # shape [N, 1]

def erode_mask(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """
    对二值 mask (0/1 tensor) 进行侵蚀操作，使得 0 区域向外扩张
    参数:
        mask: 形状 [N,1,H,W] 或 [1,1,H,W] 的二值 tensor
        kernel_size: 卷积核大小 (默认 3，相当于 3x3 邻域)
        iterations: 迭代次数，次数越多侵蚀越强
    返回:
        eroded_mask: 侵蚀后的 mask (0/1 tensor)
    """
    x = mask
    for _ in range(iterations):
        # 取反：0->1, 1->0
        inv = 1 - x
        # 对反转后的 mask 做 max pooling (相当于膨胀 0)
        pooled = Fn.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        # 再取反回来
        x = 1 - pooled
    return x
