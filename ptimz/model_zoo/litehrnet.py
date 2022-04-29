import logging
from numbers import Integral

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from mmpose.models.backbones.resnet import BasicBlock, Bottleneck, ResLayer, get_expansion
from mmpose.models.backbones.utils import load_checkpoint, channel_shuffle
from mmpose.utils import get_root_logger
from torch.nn.modules.batchnorm import _BatchNorm

from .helpers import build_model_with_cfg
from .registry import register_model

__all__ = ['LiteHRNet', 'LiteHRModule']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        # 'interpolation': 'trilinear',
        'num_classes': 1000,
        'first_conv': 'stem.conv1.conv',
        **kwargs
    }


default_cfgs = {
    "litehrnet_seg3d\tfetal_whitematter": _cfg(
        url='https://github.com/RimeT/ptimz/releases/download/v0.0.1-np/hrnetlite_feta21_wm3d.pth.tar',
        input_details='MR [T2-weighted]',
        spacing=(0.5469, 0.5469, 0.5469),
        slice_thichness=0.5469,
        num_classes=2, input_size=(1, 224, 224, 224), last_layer='head_layer.final_layer.1'),
}


def channel_shuffle3d(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, depth, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, depth, height, width)

    return x


class SpatialWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        # modified by tiansong
        if '2d' in conv_cfg['type']:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        elif '3d' in conv_cfg['type']:
            self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        elif '1d' in conv_cfg['type']:
            self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f'Unknown dimension conv_cfg {conv_cfg}')
        # end by tiansong
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class CrossResolutionWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        if '3d' in self.conv1.conv_cfg['type']:
            index = 3
        elif '2d' in self.conv1.conv_cfg['type']:
            index = 2
        elif '1d' in self.conv1.conv_cfg['type']:
            index = 1
        else:
            raise ValueError(f'Unknown conv cfg {self.conv1.conv_cfg}')
        mini_size = x[-1].size()[-index:]
        if '3d' in self.conv1.conv_cfg['type']:
            out = [F.adaptive_avg_pool3d(s, mini_size) for s in x[:-1]] + [x[-1]]
        else:
            out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-index:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.depthwise_convs = nn.ModuleList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=channel,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4,
                             # add by tiansong
                             conv_cfg=conv_cfg
                             )
            for channel in branch_channels
        ])

    def forward(self, x):

        def _inner_forward(x):
            x = [s.chunk(2, dim=1) for s in x]
            x1 = [s[0] for s in x]
            x2 = [s[1] for s in x]

            x2 = self.cross_resolution_weighting(x2)
            x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
            x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

            out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]

            if '3d' in self.depthwise_convs[0].conv_cfg['type']:
                out = [channel_shuffle3d(s, 2) for s in out]
            else:
                out = [channel_shuffle(s, 2) for s in out]

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class Stem(nn.Module):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,
                 first_stride=None,
                 branch_stride=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        first_stride = 2 if first_stride is None else first_stride
        branch_stride = 2 if branch_stride is None else branch_stride

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=first_stride,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=branch_stride,
                padding=1,
                groups=branch_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=branch_stride,
            padding=1,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            x1, x2 = x.chunk(2, dim=1)

            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)

            out = torch.cat((self.branch1(x1), x2), dim=1)
            if '3d' in self.conv1.conv_cfg['type']:
                out = channel_shuffle3d(out, 2)
            else:
                out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class ClsHead(nn.Module):

    def __init__(self, pre_stage_channels, head_cfg, conv_cfg, norm_cfg=dict(type='BN')):
        super().__init__()
        assert isinstance(head_cfg['nclasses'], Integral), "ClsHead nclasses must be Integral"
        head_block = Bottleneck
        self.conv_cfg = conv_cfg
        # TODO: beware of expansion size
        head_expansion = get_expansion(Bottleneck, None)
        head_channels = [32, 64, 128, 256]
        incre_modules = []

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        for i, channels in enumerate(pre_stage_channels):
            incre_module = ResLayer(block=head_block,
                                    num_blocks=1,
                                    in_channels=channels,
                                    out_channels=head_channels[i] * head_expansion,
                                    stride=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg
                                    )
            incre_modules.append(incre_module)
        self.incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_expansion
            out_channels = head_channels[i + 1] * head_expansion

            downsamp_modules.append(ConvModule(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               conv_cfg=conv_cfg,
                                               norm_cfg=norm_cfg,
                                               act_cfg=dict(type='ReLU', inplace=True)))
        self.downsamp_modules = nn.ModuleList(downsamp_modules)

        self.final_layer = ConvModule(in_channels=head_channels[3] * head_expansion,
                                      out_channels=2048,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=dict(type='ReLU', inplace=True))

        self.classifier = nn.Linear(2048, head_cfg['nclasses'])

    def forward(self, x_list):
        y = self.incre_modules[0](x_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](x_list[i + 1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            if '2d' in self.conv_cfg['type']:
                y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
            elif '3d' in self.conv_cfg['type']:
                y = F.avg_pool3d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
            else:
                raise ValueError(f'Cls head unsupport convtype {self.conv_cfg}')
        y = self.classifier(y)
        return y


class SegHead(nn.Module):

    def __init__(self, in_channels, head_cfg, conv_cfg, norm_cfg=dict(type='BN')):
        super().__init__()
        assert isinstance(head_cfg['nclasses'], Integral), "SegHead nclasses must be Integral"
        in_channels = int(sum(in_channels))
        final_conv_kernel = 1 if 'final_conv_kernel' not in head_cfg else head_cfg['final_conv_kernel']
        self.final_layer = nn.Sequential(*[
            ConvModule(in_channels, in_channels, kernel_size=1,
                       stride=1, padding=0, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=False)),
            build_conv_layer(conv_cfg, in_channels=in_channels, out_channels=head_cfg['nclasses'],
                             kernel_size=final_conv_kernel, stride=1,
                             padding=1 if final_conv_kernel == 3 else 0)
        ])
        if '2d' in conv_cfg['type']:
            self.interp_mode = 'bilinear'
        elif '3d' in conv_cfg['type']:
            self.interp_mode = 'trilinear'
        else:
            self.interp_mode = 'linear'

    def forward(self, x):
        # up sampling
        upsize = tuple(x[0].shape[2:])
        xs = []
        for feat_map in x[1:]:
            xs.append(F.interpolate(feat_map, size=upsize, mode=self.interp_mode))
        x = torch.cat([x[0]] + xs, 1)
        # final convolution
        x = self.final_layer(x)

        return [x]


class IterativeHead(nn.Module):

    def __init__(self, in_channels, conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        # add by tiansong
                        conv_cfg=conv_cfg,
                        # end by tiansong
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.ModuleList(projects)
        if '2d' in conv_cfg['type']:
            self.interp_mode = 'bilinear'
        elif '3d' in conv_cfg['type']:
            self.interp_mode = 'trilinear'
        else:
            self.interp_mode = 'linear'

    def forward(self, x):
        x = x[::-1]

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[2:],
                    mode=self.interp_mode,
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class ShuffleUnit(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):

        def _inner_forward(x):
            if self.stride > 1:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class LiteHRModule(nn.Module):

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            multiscale_output=False,
            with_fuse=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """Make one branch."""
        layers = []
        layers.append(
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU'),
                with_cp=self.with_cp))
        for i in range(1, num_blocks):
            layers.append(
                ShuffleUnit(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=dict(type='ReLU'),
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        if self.module_type == 'LITE':
            out = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out


# @BACKBONES.register_module()
class LiteHRNet(nn.Module):
    """Lite-HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`_

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmpose.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)

    Segmentation Example:
            extra = dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            head_type='seghead',
            head_cfg=dict(
                nclasses=3,
            )
        )
        # build net
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='SyncBN')
        net = LiteHRNet(extra, in_channels=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

    Classification Example:
            head_type='clshead',
            head_cfg=dict(nclasses=3)
    """

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        super().__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['stem']['stem_channels'],
            out_channels=self.extra['stem']['out_channels'],
            expand_ratio=self.extra['stem']['expand_ratio'],
            first_stride=self.extra['stem'].get('first_stride'),
            branch_stride=self.extra['stem'].get('branch_stride'),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            setattr(
                self, 'transition{}'.format(i),
                self._make_transition_layer(num_channels_last, num_channels))

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True)
            setattr(self, 'stage{}'.format(i), stage)

        self.head_type = self.extra.get('head_type', None)
        if 'seghead' == self.head_type:
            self.head_layer = SegHead(num_channels_last,
                                      head_cfg=self.extra['head_cfg'],
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg)
        elif 'posehead' == self.head_type:
            self.head_layer = IterativeHead(
                in_channels=num_channels_last,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )
        elif 'clshead' == self.head_type:
            self.head_layer = ClsHead(
                pre_stage_channels=num_channels_last,
                head_cfg=self.extra['head_cfg'],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_pre_layer[i])[1],
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels)[1],
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    stage_index,
                    in_channels,
                    multiscale_output=True):
        num_modules = stages_spec['num_modules'][stage_index]
        num_branches = stages_spec['num_branches'][stage_index]
        num_blocks = stages_spec['num_blocks'][stage_index]
        reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]
        module_type = stages_spec['module_type'][stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))
            in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        origin_shape = x.shape
        x = self.stem(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i))
            for j in range(self.stages_spec['num_branches'][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i))(x_list)

        x = y_list
        if self.head_type is not None:
            x = self.head_layer(x)
            if 'seghead' == self.head_type:
                if '2d' in self.conv_cfg['type']:
                    interp_mode = 'bilinear'
                elif '3d' in self.conv_cfg['type']:
                    interp_mode = 'trilinear'
                else:
                    interp_mode = 'linear'
                for idx, y in enumerate(x):
                    x[idx] = F.interpolate(y, tuple(origin_shape[2:]), mode=interp_mode)
                return x[0]
            elif 'posehead' == self.head_type:
                return [x[0]]
            elif 'clshead' == self.head_type:
                return x

        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def _build_hrnet(pretrained_name, head_type, dimension="3d", in_chans=1, num_classes=1000, **kwargs):
    dimension = dimension.lower()
    assert dimension in ('2d', '3d'), "dimension must be 2d or 3d"
    head_dict = dict(nclasses=num_classes)
    extra = dict(
        stem=dict(
            stem_channels=32,
            first_stride=2,
            branch_stride=2,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(3, 8, 3),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320),
            )),
        head_type=head_type,
        head_cfg=head_dict
    )
    conv_cfg = dict(type=f'Conv{dimension}')
    norm_cfg = dict(type=f'BN{dimension}')

    # direct invoke
    # model = LiteHRNet(extra, in_channels=in_chans, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
    # return model

    # use cfg to build
    pretrained = False if pretrained_name is False or pretrained_name is None else True
    return build_model_with_cfg(model_cls=LiteHRNet, variant='litehrnet', pretrained=pretrained,
                                default_cfg=default_cfgs.get(pretrained_name, None),
                                # model config
                                extra=extra, in_channels=in_chans, conv_cfg=conv_cfg, norm_cfg=norm_cfg)


@register_model
def litehrnet_seg3d(pretrained=None, **kwargs):
    if pretrained is True:
        pretrained = 'litehrnet_seg3d\tfetal_whitematter'
    elif isinstance(pretrained, str):
        pretrained = 'litehrnet_seg3d' + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger = logging.getLogger(__name__)
            _logger.warning(f"There is no {pretrained} pretrained weights, use random initialization.")
            pretrained = False

    return _build_hrnet(pretrained, 'seghead', '3d', **kwargs)


@register_model
def litehrnet_cls3d(pretrained=None, **kwargs):
    return _build_hrnet(pretrained, 'clshead', '3d', **kwargs)


@register_model
def litehrnet_seg2d(pretrained=None, **kwargs):
    return _build_hrnet(pretrained, 'seghead', '2d', **kwargs)


@register_model
def litehrnet_cls2d(pretrained=None, **kwargs):
    return _build_hrnet(pretrained, 'clshead', '2d', **kwargs)
