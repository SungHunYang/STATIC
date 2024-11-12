import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from einops import rearrange
from einops.layers.torch import Rearrange
import math

class Sobel(nn.Module):
    def __init__(self,filter):
        super(Sobel, self).__init__()
        self.filter = filter

    def made_filter(self):
        if self.filter == 5:
            self.sobel_x = torch.tensor([[2, 1, 0, -1, -2],
                                         [3, 2, 0, -2, -3],
                                         [4, 3, 0, -3, -4],
                                         [3, 2, 0, -2, -3],
                                         [2, 1, 0, -1, -2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            self.sobel_y = torch.tensor([[2, 3, 4, 3, 2],
                                         [1, 2, 3, 2, 1],
                                         [0, 0, 0, 0, 0],
                                         [-1, -2, -3, -2, -1],
                                         [-2, -3, -4, -3, -2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(
                0).unsqueeze(
                0)
            self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(
                0).unsqueeze(
                0)

        return self.sobel_x, self.sobel_y

    def compute_normals_from_depth(self, depth):
        B, C, H, W = depth.size()
        assert C == 1, "Depth image should have a single channel"

        sobel_x, sobel_y = self.made_filter()
        sobel_x = sobel_x.to(depth.device)
        sobel_y = sobel_y.to(depth.device)

        if self.filter == 5:
            grad_x = F.conv2d(depth, sobel_x, padding=2)
            grad_y = F.conv2d(depth, sobel_y, padding=2)
        else:
            grad_x = F.conv2d(depth, sobel_x, padding=1)
            grad_y = F.conv2d(depth, sobel_y, padding=1)

        grad_z = torch.ones_like(grad_x)
        normals = torch.cat((grad_x.contiguous(), grad_y.contiguous(), grad_z.contiguous()), dim=1)
        normals = F.normalize(normals, p=2, dim=1)

        return normals

    def compute_normals_from_gt(self, depth):
        B, C, H, W = depth.size()
        assert C == 1, "Depth image should have a single channel"

        sobel_x, sobel_y = self.made_filter()
        sobel_x = sobel_x.to(depth.device)
        sobel_y = sobel_y.to(depth.device)

        if self.filter == 3:
            padding = 1
        elif self.filter == 5:
            padding = 2
        else:
            padding = 3

        # 커널 크기 계산
        kernel_size = sobel_x.size(2)

        # Depth 값이 0인 위치 찾기
        zero_mask = (depth == 0).float()

        # Zero mask를 dilation 연산을 통해 확장 (상하좌우 1픽셀 고려)
        expanded_zero_mask = F.max_pool2d(zero_mask, kernel_size=5, stride=1, padding=2) # (3,1) -> 1 픽셀, (5,2)-> 2픽셀,

        # Sobel 필터 적용
        grad_x = F.conv2d(depth, sobel_x, padding=padding)
        grad_y = F.conv2d(depth, sobel_y, padding=padding)

        # Depth 값이 0인 위치의 gradient를 0으로 설정
        grad_x[expanded_zero_mask > 0] = 0
        grad_y[expanded_zero_mask > 0] = 0

        grad_z = torch.ones_like(grad_x)
        grad_z[expanded_zero_mask > 0] = 0

        normals = torch.cat((grad_x.contiguous(), grad_y.contiguous(), grad_z.contiguous()), dim=1)
        normals = F.normalize(normals, p=2, dim=1)

        return normals

    def cosine_similarity(self,norm):
        # A와 B는 각각 (b, c, h, w) 형태의 텐서

        A = norm[:,0,::]
        B = norm[:,1,::]
        C = norm[:,2,::]
        sim_mean = []
        dot_product = torch.sum(A * B, dim=1)  # c축을 따라 내적 계산
        norm_A = torch.sqrt(torch.sum(A ** 2, dim=1))
        norm_B = torch.sqrt(torch.sum(B ** 2, dim=1))
        cos_sim_1 = dot_product / (norm_A * norm_B)
        cos_sim_1 = cos_sim_1.unsqueeze(1).contiguous()
        contin_mask_1 = (cos_sim_1 >= torch.mean(cos_sim_1)).float()
        dis_mask_1 = (cos_sim_1 < torch.mean(cos_sim_1)).float()

        dot_product = torch.sum(B * C, dim=1)  # c축을 따라 내적 계산
        norm_B = torch.sqrt(torch.sum(B ** 2, dim=1))
        norm_C = torch.sqrt(torch.sum(C ** 2, dim=1))
        cos_sim_2 = dot_product / (norm_B * norm_C)
        sim_mean.append(torch.mean(cos_sim_2))
        cos_sim_2 = cos_sim_2.unsqueeze(1).contiguous()
        contin_mask_2 = (cos_sim_2 >= torch.mean(cos_sim_2)).float()
        dis_mask_2 = (cos_sim_2 < torch.mean(cos_sim_2)).float()

        cos_sim = torch.cat((cos_sim_1, cos_sim_2),dim=1).unsqueeze(dim=2).contiguous()
        continue_sim_mask = torch.cat((contin_mask_1, contin_mask_2), dim=1).unsqueeze(dim=2).contiguous()
        discrete_sim_mask = torch.cat((dis_mask_1, dis_mask_2), dim=1).unsqueeze(dim=2).contiguous()



        # return cos_sim,discrete_sim_mask,continue_sim_mask
        return {'cos_sim':cos_sim,'dis_mask':discrete_sim_mask,'con_mask':continue_sim_mask }

    def forward(self,depth,t):

        norm = self.compute_normals_from_depth(depth)
        norm = rearrange(norm, '(b t) c h w -> b t c h w', t=t)
        depth = rearrange(depth, '(b t) c h w -> b c t h w', t=t)
        norm_info = self.cosine_similarity(norm)

        return depth, norm, norm_info

    
def warp_image(image, flow):
    """
    Warp an image using the provided optical flow.

    Parameters:
    img (torch.Tensor): Source image of shape (B, C, H, W)
    flow (torch.Tensor): Optical flow of shape (B, 2, H, W)

    Returns:
    torch.Tensor: Warped image of shape (B, C, H, W)
    """
    img = image.clone().detach()
    B, C, H, W = img.size()
    flow_u = flow[:, 0, :, :].clone().detach()  # x-axis flow
    flow_v = flow[:, 1, :, :].clone().detach()  # y-axis flow

    # Generate a grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.to(img.device).float()
    grid_y = grid_y.to(img.device).float()

    # Normalize grid to [-1, 1]
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0

    # Add flow to grid
    grid_x = grid_x.unsqueeze(0) + 2.0 * flow_u / (W - 1)
    grid_y = grid_y.unsqueeze(0) + 2.0 * flow_v / (H - 1)

    grid = torch.stack((grid_x, grid_y), dim=3)

    # Use grid_sample to warp the image
    warped_img = F.grid_sample(img, grid, align_corners=True)

    return warped_img