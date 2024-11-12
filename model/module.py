import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .attention import *
import torchvision.transforms as T
from .deform import *

class NormalEncoder(nn.Module):
    def __init__(self, in_channels, embed_channel):
        super(NormalEncoder, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels[0], embed_channel, 1)
        self.conv_2 = nn.Conv2d(in_channels[1], embed_channel, 1)
        self.conv_3 = nn.Conv2d(in_channels[2], embed_channel, 1)
        self.conv_4 = nn.Conv2d(in_channels[3], embed_channel, 1)

        self.resize_3 = nn.ConvTranspose2d(
            in_channels=embed_channel,
            out_channels=embed_channel,
            kernel_size=2,
            stride=2,
            padding=0)
        self.resize_2 = nn.ConvTranspose2d(
            in_channels=embed_channel,
            out_channels=embed_channel,
            kernel_size=2,
            stride=2,
            padding=0)
        # self.resize_1 = nn.ConvTranspose2d(
        #     in_channels=embed_channel,
        #     out_channels=embed_channel,
        #     kernel_size=2,
        #     stride=2,
        #     padding=0)

        self.fusion1 = ResBlock(embed_channel, embed_channel)
        self.fusion2 = ResBlock(embed_channel, embed_channel)
        self.fusion3 = ResBlock(embed_channel, embed_channel)

        self.refine = nn.Conv2d(embed_channel, embed_channel, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(embed_channel // 4, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x1 = self.conv_1(x1)
        x2 = self.conv_2(x2)
        x3 = self.conv_3(x3)
        x4 = self.conv_4(x4)

        x3 = self.fusion1(x4, x3)
        x3 = self.resize_3(x3)
        x2 = self.fusion2(x3, x2)
        x2 = self.resize_2(x2)
        x1 = self.fusion3(x2, x1)
        # x1 = rearrange(x1,'(b t) c h w -> b c t h w',t=2)
        x1 = self.refine(x1)
        # x1 = rearrange(x1, 'b c t h w -> (b t) c h w')
        # out = self.resize_1(x1)
        out = nn.PixelShuffle(2)(x1)
        out = self.refine2(out)
        out = self.tanh(out)

        return out, x1

class ResBlock(nn.Module):
    def __init__(self, dim, outdim):
        super().__init__()

        self.dim = dim
        self.outdim = outdim

        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.dim, self.outdim, kernel_size=3, stride=1, padding=1, bias=True)
        self.fine = nn.Conv2d(self.outdim, self.outdim, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        x1 = self.act(self.conv1(x1))
        x = x1 + x2
        x = self.act(self.conv2(x))
        x = self.fine(x)
        return x

class VideoNorm(nn.Module):
    def __init__(self,features,head,dep):
        super().__init__()

        self.head = head
        self.dep = dep
        self.down1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1)*1e-6).to(torch.float32)

        self.up_info = nn.Sequential(
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2, padding=0),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
        )
        self.mask_gen = nn.Sequential(
            nn.Conv2d(features, features // 4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 4, features // 8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            Rearrange('(b t) c h w -> b c t h w', t=2),
            nn.Conv3d(features // 8, features // 8, kernel_size=3, stride=1, padding =1),
            nn.GELU(),
            nn.Conv3d(features // 8, 2, kernel_size=(2,3,3), stride=1, padding=(0,1,1)),
            nn.Softmax(dim=1),
            Rearrange('b c t h w -> (b t) c h w'),
        )

        self.f_atten = nn.ModuleList([
            Swin_Frame_Attention(features, self.head, qkv_bias=True) for i in range(self.dep * 2)
        ])
        self.d_atten = nn.ModuleList([
            Swin_Align_Attention(features, self.head, qkv_bias=True) for i in range(self.dep)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def make_mask(self, norm_feat):
        b, _, t, h, w = norm_feat.shape  # b 3 t h w

        norm_feat = (norm_feat + 1.) / 2.  # 0~1 normalize

        norm_feat = rearrange(norm_feat, 'b c t h w -> b t h w c')
        norm_var = torch.var(norm_feat, dim=1, unbiased=True)  # variance of vectors
        norm_var_magnitude = torch.norm(norm_var, dim=-1)  # (b, h, w)
        norm_mask = norm_var_magnitude.unsqueeze(1)  # (b, 1, h, w)
        norm_mean = torch.mean(norm_mask, dim=[2, 3], keepdim=True)
        norm_mask = norm_mask - (norm_mean + abs(self.alpha))
        norm_mask = torch.where(norm_mask > 0, norm_mask, torch.zeros_like(norm_mask))
        check_mask = torch.where(norm_mask == 0, torch.ones_like(norm_mask) * 1e-10, norm_mask)
        norm_mask = norm_mask / check_mask
        return norm_mask

    def prepare_mask(self,norm_feat):

        b, _, t, h, w = norm_feat.shape
        mask = self.make_mask(norm_feat)

        return mask

    def forward(self, depth, normal, c3, x):

        b,c,t,h,w = normal.size()
        normal = normal.detach().clone()

        mask = self.prepare_mask(normal)
        gen = c3.detach().clone() + x
        gen = self.up(gen)
        gen_mask = self.mask_gen(gen)

        depth = F.interpolate(depth, size=(h // 4, w // 4), mode='bilinear', align_corners=True) # 이게 절반으로 줄인거
        depth = self.down1(depth)
        c3 = F.interpolate(c3, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        c3 = self.down2(c3)

        depth = rearrange(depth, '(b t) c h w -> b c t h w', t=t)
        c3 = rearrange(c3, '(b t) c h w -> b c t h w', t=t)
        n_mask = F.interpolate(gen_mask, (h // 4, w // 4), mode='bilinear', align_corners=True)
        n_mask = torch.cat((n_mask.unsqueeze(2), n_mask.unsqueeze(2)), dim=2).contiguous() # b 2 2 h w

        # static
        static = depth * n_mask[:,1,:,:,:].unsqueeze(dim=1)
        for i, blk in enumerate(self.f_atten):
            static = blk(static, n_mask)
            static = static * n_mask[:,1,:,:,:].unsqueeze(dim=1)

        # dynamic
        dynamic = depth
        for i, blk in enumerate(self.d_atten):
            dynamic = blk(dynamic, c3, n_mask)

        # cat dynamic, continuous
        video = dynamic * n_mask[:,0,:,:,:].unsqueeze(dim=1) + static # change
        video = rearrange(video, 'b c t h w -> (b t) c h w')
        video = self.up_info(video)

        return video, mask, gen_mask
