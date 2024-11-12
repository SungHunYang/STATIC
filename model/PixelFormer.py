import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .SAM import SAM
from timm.models.layers import trunc_normal_
########################################################################################################################

class my_PQI(nn.Module):
    def __init__(self, dim, outdim, padding_size=(1,2,4,6)):
        super(my_PQI, self).__init__()

        self.padding_size = padding_size

        embed = outdim // (len(padding_size))

        self.iden = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(dim, embed, kernel_size=1, padding=0, bias=False),
        )
        self.adap1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(padding_size[1]),
            nn.Conv2d(dim, embed, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed, embed, kernel_size=1, padding=0, bias=False),
        )
        self.adap2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(padding_size[2]),
            nn.Conv2d(dim, embed, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed, embed, kernel_size=1, padding=0, bias=False),
        )
        self.adap3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(padding_size[3]),
            nn.Conv2d(dim, embed, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed, embed, kernel_size=1, padding=0, bias=False),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(embed * len(padding_size), outdim, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(outdim, outdim),
            nn.GELU(),
            nn.Conv2d(outdim, outdim, kernel_size=1, padding=0, bias=False)
        )


    def forward(self, x):
        b,c,h,w = x.shape

        x1 = self.iden(x)
        x2 = self.adap1(x)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)
        x3 = self.adap2(x)
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=True)
        x4 = self.adap3(x)
        x4 = F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)

        x_out = torch.cat((x1, x2, x3, x4), dim=1).contiguous()
        x = self.conv(x_out)

        return x



class BCP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, max_depth, min_depth, in_features=512, hidden_features=512*4, out_features=256, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        x = torch.mean(x.flatten(start_dim=2), dim = 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1).contiguous()
        return centers

class PixelFormer(nn.Module):

    def __init__(self, in_channels, num_heads, inv_depth=False, pretrained=None,
                    frozen_stages=-1, min_depth=1e-3, max_depth=80.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        self.num_heads = num_heads

        win = 16 # 12
        sam_dims = [in_channels[0], 384, 768, 768]
        v_dims = [96, 192, 768, 768]
        self.sam4 = SAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=win, v_dim=v_dims[3], num_heads=24)
        self.sam3 = SAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.sam2 = SAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.sam1 = SAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)


        self.decoder = my_PQI(in_channels[3], 768)

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


    def forward(self, enc_feats):

        q4 = self.decoder(enc_feats[3])
        q3 = self.sam4(enc_feats[3], q4)
        q2 = self.sam3(enc_feats[2], q3)
        q2 = nn.PixelShuffle(2)(q2)
        q1 = self.sam2(enc_feats[1], q2)
        q1 = nn.PixelShuffle(2)(q1)
        q0 = self.sam1(enc_feats[0], q1)

        return q0


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, centers, scale):
        x = self.conv1(x)
        x = x.softmax(dim=1)
        x = torch.sum(x * centers, dim=1, keepdim=True)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)