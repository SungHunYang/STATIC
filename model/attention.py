from torch import Tensor
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch
from .swin_block import SAM
from .swin_block_v2 import SAMv2
from .swin_block_v3 import SAMv3

class Swin_Reg_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            patch=2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.embed = nn.Conv2d(dim, dim, 3, 2, 1)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, 2, 0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )
        win = 7
        self.sam = SAM(input_dim=dim, embed_dim=dim, window_size=win, v_dim=dim, num_heads=self.num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):

        x = self.embed(x)
        out = self.sam(x, x)
        out = self.up(out)

        return out

class Swin_Frame_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            patch=2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        sr_ratio = 4
        self.patch = patch
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.proj_drop = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(dim)

        win = 12
        self.sam = SAMv3(input_dim=dim, embed_dim=dim, window_size=win, v_dim=dim, num_heads=self.num_heads)
        self.sam2 = SAMv3(input_dim=dim, embed_dim=dim, window_size=win, v_dim=dim, num_heads=self.num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x, mask):
        B, c, t, h, w = x.shape

        x = rearrange(x, 'b c t h w -> b t c h w')
        frame_1 = x[:, 0, ::]
        frame_2 = x[:, 1, ::]
        mask = mask[:, 1, 0, :, :].unsqueeze(dim=1)

        out = self.sam(frame_1, frame_2, mask)
        x1 = self.proj_1(out)

        out = self.sam2(frame_2, frame_1, mask)
        x2 = self.proj_2(out)

        x = torch.cat([x1.unsqueeze(dim=2), x2.unsqueeze(dim=2)], dim=2).contiguous()

        return x

class Swin_Bin_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            patch=2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.embed = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
        )
        # self.embed = nn.Conv2d(dim, dim, kernel_size=3, stride=2,padding=1)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.ConvTranspose2d(dim * 2, dim * 2, 2, 2, 0),
        )
        win = 16
        self.sam = SAM(input_dim=dim * 2, embed_dim=dim * 2, window_size=win, v_dim=dim * 2, num_heads=self.num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):

        x = F.interpolate(x, scale_factor = 0.25, mode='bilinear', align_corners=True)
        x = self.embed(x)
        out = self.sam(x, x)
        out = self.up(out)

        return out


class Swin_Self_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            patch=2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.embed = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        # self.embed = nn.Conv2d(dim, dim, kernel_size=3, stride=2,padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, 2, 2, 0)
        )
        win = 16
        self.sam = SAM(input_dim=dim, embed_dim=dim, window_size=win, v_dim=dim, num_heads=self.num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):

        x = F.interpolate(x, scale_factor = 0.5, mode='bilinear', align_corners=True)
        x = self.embed(x)
        out = self.sam(x, x)
        out = self.up(out)

        return out


class Swin_Align_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            patch=2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.patch = patch
        self.layer_norm = nn.LayerNorm(dim)

        win = 16
        self.sam = SAMv2(input_dim=dim, embed_dim=dim, window_size=win, v_dim=dim, num_heads=self.num_heads)
        self.sam2 = SAMv2(input_dim=dim, embed_dim=dim, window_size=win, v_dim=dim, num_heads=self.num_heads)

        self.x1 = nn.Conv2d(dim * 2, dim, 3, 1, 1)

        self.x2 = nn.Conv2d(dim * 2, dim, 3, 1, 1)

        self.fine_norm = nn.Conv2d(dim, dim, 3, 1, 1)
        self.fine_norm2 = nn.Conv2d(dim, dim, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x, norm, mask):
        B, c, t, h, w = x.shape

        dep1 = x[:, :, 0, :, :]
        dep2 = x[:, :, 1, :, :]
        norm1 = norm[:, :, 0, :, :]
        norm2 = norm[:, :, 1, :, :]
        frame1 = torch.cat([norm1, dep1], dim=1).contiguous()
        frame2 = torch.cat([norm2, dep2], dim=1).contiguous()
        mask = mask[:, 0, 0, :, :].unsqueeze(dim=1)

        out = self.sam(frame1, frame2, dep2, dep1, mask)  # q k v skip mask
        norm1 = self.fine_norm(norm1)
        x1 = torch.cat([out, norm1 * mask], dim=1)
        x1 = self.x1(x1) + out

        out = self.sam2(frame2, frame1, dep1, dep2, mask)  # q k v skip mask
        norm2 = self.fine_norm2(norm2)
        x2 = torch.cat([out, norm2 * mask], dim=1)
        x2 = self.x2(x2) + out

        x = torch.cat([x1.unsqueeze(dim=2), x2.unsqueeze(dim=2)], dim=2).contiguous()

        return x