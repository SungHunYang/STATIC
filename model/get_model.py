from .swin_transformer_v2 import SwinTransformerV2
from .PixelFormer import *
from .module import *
import sys
from . import model_config as config
import torch
import torch.nn as nn
from einops import rearrange
import time
from .attention import *
from .swin_block_v4 import SAMv4

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.GroupNorm(out_channels,out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.GroupNorm(out_channels,out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.GroupNorm(out_channels,out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.GroupNorm(out_channels,out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.GroupNorm(out_channels,out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.GroupNorm(out_channels, out_channels)
        self.conv_1x1_4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        # out_1x1 = F.gelu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        # out_3x3_1 = F.gelu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        # out_3x3_2 = F.gelu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        # out_3x3_3 = F.gelu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
        out_1x1 = self.act((self.conv_1x1_1(feature_map)))
        out_3x3_1 = self.act(self.conv_3x3_1(feature_map))
        out_3x3_2 = self.act(self.conv_3x3_2(feature_map))
        out_3x3_3 = self.act(self.conv_3x3_3(feature_map))

        out_img = self.avg_pool(feature_map)
        out_img = self.act(self.conv_1x1_2(out_img))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1).contiguous()
        out = self.act(self.conv_1x1_3(out))
        out = self.conv_1x1_4(out)

        return out

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0

    for name, param in state_dict.items():
        name = name.replace('encoder.', '')
        if name not in list(own_state.keys()):
            ckpt_name.append(name)
        else:
            try:
                own_state[name].copy_(param)
                cnt += 1
            except:
                continue

    print('#reused param : {} / {}\n'.format(cnt, len(state_dict.items())))

    return model

class build_model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        if config.TYPE == 'swinv2':
            self.encoder = SwinTransformerV2(img_size=config.MODEL[config.MODE]['IMG_SIZE'],
                                      patch_size=config.CONF['PATCH_SIZE'],
                                      in_chans=config.CONF['IN_CHANS'],
                                      num_classes=0,
                                      embed_dim=config.MODEL[config.MODE]['EMBED_DIM'],
                                      depths=config.MODEL[config.MODE]['DEPTHS'],
                                      num_heads=config.MODEL[config.MODE]['NUM_HEADS'],
                                      window_size=config.MODEL[config.MODE]['WINDOW_SIZE'],
                                      mlp_ratio=config.CONF['MLP_RATIO'],
                                      qkv_bias=config.CONF['QKV_BIAS'],
                                      drop_rate=config.CONF['DROP_RATE'],
                                      drop_path_rate=config.MODEL[config.MODE]['DROP_PATH_RATE'],
                                      ape=config.CONF['APE'],
                                      patch_norm=config.CONF['PATCH_NORM'],
                                      use_checkpoint=False,
                                      pretrained_window_sizes=config.CONF['PRETRAINED_WINDOW_SIZES'],
                                             )
        self.head = {
            's': 4, 'b': 8, 'l': 12
        }
        self.dep = {
            's': 1, 'b': 1, 'l': 1
        }
        embed_dim = config.MODEL[config.MODE]['EMBED_DIM']
        in_chans = [embed_dim*2, embed_dim*4, embed_dim*8, embed_dim*8]
        self.dim = in_chans[0]

        self.decoder = PixelFormer(in_chans,config.MODEL[config.MODE]['NUM_HEADS'])
        self.norm_encoder = NormalEncoder(in_chans,self.dim)
        self.video = VideoNorm(self.dim,self.head[config.MODE],self.dep[config.MODE])

        self.aspp = ASPP(self.dim, self.dim)
        # self.up_info = nn.Conv2d(self.dim // 4, self.dim // 2, kernel_size=5,stride=1,padding=2)
        self.up_info = nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim, kernel_size=2,stride=2,padding=0),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim // 2, kernel_size=3,stride=1,padding=1),
        )

        # self.add = nn.Sequential(
        #     nn.Conv2d(self.dim, self.dim,1,1,0),
        #     nn.GELU(),
        #     nn.Conv2d(self.dim, self.dim,1,1,0),
        # )

        win = 16
        self.sam = SAMv4(input_dim=self.dim, embed_dim=self.dim, window_size=win, v_dim=self.dim,
                         num_heads=4)

        # self.s_atten = Swin_Reg_Attention(self.dim // 4, self.head[config.MODE] // 3, qkv_bias=True)

        self.last = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dim // 2, self.dim // 4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(self.dim // 4, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        if self.pretrained:
            model_path = config.PRETRAIN[config.MODE]
            # checkpoint = torch.load(model_path,map_location='cpu')['model']
            checkpoint = torch.load(model_path,weights_only=False,map_location='cpu')['model']
            self.encoder = load_my_state_dict(self.encoder,checkpoint)
            print('============ load encoder pretrained model ============')

    def forward(self, x):

        b,t,c,h,w = x.size()
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.encoder(x)
        x[0] = rearrange(x[0], 'b (h w) c -> b c h w',h=h//8,w=w//8)
        skip = x[0].detach().clone()
        x[1] = rearrange(x[1],'b (h w) c -> b c h w',h=h//16,w=w//16)
        x[2] = rearrange(x[2], 'b (h w) c -> b c h w',h=h//32,w=w//32)
        x[3] = rearrange(x[3], 'b (h w) c -> b c h w',h=h//32,w=w//32)

        norm, c3 = self.norm_encoder(x) # /8
        norm = rearrange(norm,'(b t) c h w -> b c t h w',t=t)

        x = self.decoder(x) # /8
        x = self.aspp(x) + x

        video, mask, gen_mask = self.video(x, norm, c3, skip)
        norm = rearrange(norm, 'b c t h w -> b t c h w')

        depth = self.sam(x, video)
        # depth = nn.PixelShuffle(2)(depth)

        # depth = self.s_atten(depth)
        depth = self.up_info(depth)
        depth = self.last(depth) * 80.0

        depth = rearrange(depth, '(b t) c h w -> b t c h w', t=t)

        if self.pretrained:
            return depth, norm, mask, gen_mask
        else:
            return depth, norm
