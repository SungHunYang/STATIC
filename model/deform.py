import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 first=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.first = first
        if first:
            self.offset_conv = nn.Conv3d(in_channels + 1,
                                         2 * kernel_size[0] * kernel_size[1],
                                         kernel_size=kernel_size[0],
                                         stride=(1,stride,stride),
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         bias=True)
            self.modulator_conv = nn.Conv3d(in_channels + 1,
                                            1 * kernel_size[0] * kernel_size[1],
                                            kernel_size=kernel_size[0],
                                            stride=(1,stride,stride),
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=True)
            self.regular_conv_0 = nn.Conv2d(in_channels=in_channels + 1,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)
            self.regular_conv_1 = nn.Conv2d(in_channels=in_channels + 1,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)
        else:
            self.offset_conv = nn.Conv3d(in_channels,
                                         2 * kernel_size[0] * kernel_size[1],
                                         kernel_size=kernel_size[0],
                                         stride=(1,stride,stride),
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         bias=True)
            self.modulator_conv = nn.Conv3d(in_channels,
                                            1 * kernel_size[0] * kernel_size[1],
                                            kernel_size=kernel_size[0],
                                            stride=(1,stride,stride),
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=True)
            self.regular_conv_0 = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)
            self.regular_conv_1 = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.
        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset) # b c t h w
        modulator = 2. * torch.sigmoid(self.modulator_conv(x)) # b c t h w
        # op = (n - (k * d - 1) + 2p / s)
        if self.first:
            x_0 = torchvision.ops.deform_conv2d(input=x[:, :, 0, :, :],
                                                offset=offset[:, :, 0, :, :],
                                                weight=self.regular_conv_0.weight,
                                                bias=self.regular_conv_0.bias,
                                                padding=self.padding,
                                                mask=modulator[:, :, 0, :, :],
                                                stride=self.stride,
                                                dilation=self.dilation)

            x_1 = torchvision.ops.deform_conv2d(input=x[:, :, 1, :, :],
                                                offset=offset[:, :, 1, :, :],
                                                weight=self.regular_conv_0.weight,
                                                bias=self.regular_conv_1.bias,
                                                padding=self.padding,
                                                mask=modulator[:, :, 1, :, :],
                                                stride=self.stride,
                                                dilation=self.dilation)
        else:
            x_0 = torchvision.ops.deform_conv2d(input=x[:, :, 0, :, :],
                                                offset=offset[:, :, 0, :, :],
                                                weight=self.regular_conv_0.weight,
                                                bias=self.regular_conv_0.bias,
                                                padding=self.padding,
                                                mask=modulator[:, :, 0, :, :],
                                                stride=self.stride,
                                                dilation=self.dilation)

            x_1 = torchvision.ops.deform_conv2d(input=x[:, :, 1, :, :],
                                                offset=offset[:, :, 1, :, :],
                                                weight=self.regular_conv_0.weight,
                                                bias=self.regular_conv_1.bias,
                                                padding=self.padding,
                                                mask=modulator[:, :, 1, :, :],
                                                stride=self.stride,
                                                dilation=self.dilation)

        x = torch.cat((x_0.unsqueeze(2), x_1.unsqueeze(2)), dim=2)

        return x


class DeformableConv2d2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 first=False):
        super(DeformableConv2d2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.first = first
        if first:
            self.offset_conv = nn.Conv2d(in_channels + 1,
                                         2 * kernel_size[0] * kernel_size[1],
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         bias=True)
            self.modulator_conv = nn.Conv2d(in_channels + 1,
                                            1 * kernel_size[0] * kernel_size[1],
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=True)
            self.regular_conv_0 = nn.Conv2d(in_channels=in_channels + 1,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)
            self.regular_conv_1 = nn.Conv2d(in_channels=in_channels + 1,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)
        else:
            self.offset_conv = nn.Conv2d(in_channels,
                                         2 * kernel_size[0] * kernel_size[1],
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         bias=True)
            self.modulator_conv = nn.Conv2d(in_channels,
                                            1 * kernel_size[0] * kernel_size[1],
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=True)
            self.regular_conv_0 = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)
            self.regular_conv_1 = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=bias)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.
        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset) # b c t h w
        modulator = 2. * torch.sigmoid(self.modulator_conv(x)) # b c t h w
        # op = (n - (k * d - 1) + 2p / s)
        if self.first:
            x = torchvision.ops.deform_conv2d(input=x,
                                                offset=offset,
                                                weight=self.regular_conv_0.weight,
                                                bias=self.regular_conv_0.bias,
                                                padding=self.padding,
                                                mask=modulator,
                                                stride=self.stride,
                                                dilation=self.dilation)

        else:
            x = torchvision.ops.deform_conv2d(input=x,
                                              offset=offset,
                                              weight=self.regular_conv_0.weight,
                                              bias=self.regular_conv_0.bias,
                                              padding=self.padding,
                                              mask=modulator,
                                              stride=self.stride,
                                              dilation=self.dilation)



        return x