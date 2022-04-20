import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from mmpose.models.backbones.utils import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ptimz.model_zoo.resnet import BasicBlock, Bottleneck, ResNet
from .helpers import build_model_with_cfg
from .registry import register_model

__all__ = ['UNet']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        # 'interpolation': 'trilinear',
        'num_classes': 1000,
        **kwargs
    }


default_cfgs = {
    "resunet50_2d\tmultiplesclerosis": _cfg(
        url='https://github.com/RimeT/ptimz/releases/download/v0.0.1-np/resunet50_2d_multiplesclerosis_FLAIR.pth.tar',
        input_details='MR [FLAIR]',
        spacing=(0.5, 0.5),
        slice_thickness=5,
        first_conv='encoder_0.0.conv',
        num_classes=2, input_size=(1, 512, 512), last_layer='head_layer.final_layer.1'),
}


def dropout_layer(conv_cfg, dropout_rate=0):
    if '2d' in conv_cfg['type']:
        dropout = nn.Dropout2d(dropout_rate)
    elif '3d' in conv_cfg['type']:
        dropout = nn.Dropout3d(dropout_rate)
    else:
        dropout = nn.Dropout(dropout_rate)
    return dropout


class UpBlock(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block=Bottleneck,
                 up_sampling=True,
                 up_type='deconv',
                 conv_cfg=None,
                 dropout_rate=0.0,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        if '2d' in conv_cfg['type']:
            self.interp_mode = 'bilinear'
        elif '3d' in conv_cfg['type']:
            self.interp_mode = 'trilinear'
        else:
            self.interp_mode = 'linear'
        if up_sampling:
            if 'deconv' == up_type:
                if '2d' in conv_cfg['type']:
                    self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=2,
                                                 stride=2)
                elif '3d' in conv_cfg['type']:
                    self.up = nn.ConvTranspose3d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=2,
                                                 stride=2)
                elif '1d' in conv_cfg['type']:
                    self.up = nn.ConvTranspose1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=2,
                                                 stride=2)
            elif 'interp' == up_type:
                self.up = nn.Upsample(scale_factor=2, mode=self.interp_mode, align_corners=False)
        else:
            self.up = None

        self.dropout = dropout_layer(conv_cfg, dropout_rate)
        self.conv = block(in_channels=out_channels * 2, out_channels=out_channels, conv_cfg=conv_cfg,
                          norm_cfg=norm_cfg,
                          downsample=nn.Sequential(build_conv_layer(conv_cfg,
                                                                    in_channels=out_channels * 2,
                                                                    out_channels=out_channels,
                                                                    kernel_size=1,
                                                                    stride=1,
                                                                    padding=0),
                                                   build_norm_layer(norm_cfg, out_channels)[1]))

    def forward(self, input1, input2, *args):
        if self.up is not None:
            input1 = self.up(input1)
        # hard up
        if input1.shape[2:] != input2.shape[2:]:
            input1 = F.interpolate(input1, size=input2.shape[2:], mode=self.interp_mode, align_corners=False)
        x = torch.cat((input1, input2), dim=1)
        x = self.conv(self.dropout(x))
        return x


class UNet(nn.Module):
    def __init__(self,
                 extra,
                 layers,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 upsample_type='deconv',
                 decoder_block=Bottleneck,
                 dropout_rate=0,
                 zero_init_residual=True,
                 aux_train=False,
                 **kwargs) -> None:

        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.extra = extra
        self.zero_init_residual = zero_init_residual
        self.aux_train = aux_train
        self.down_layer_name = []
        for encoder_id, encoder_layer in enumerate(layers):
            self.add_module(f"encoder_{encoder_id}", encoder_layer)
            self.down_layer_name.append(f"encoder_{encoder_id}")
        self.up_layer_name = []
        num_channels = self.extra.get('num_channels')
        for up_id in range(len(num_channels) - 1):
            self.add_module(f"decoder_{up_id}", UpBlock(in_channels=num_channels[up_id + 1],
                                                        out_channels=num_channels[up_id],
                                                        block=decoder_block,
                                                        conv_cfg=conv_cfg,
                                                        norm_cfg=norm_cfg,
                                                        up_type=upsample_type,
                                                        dropout_rate=dropout_rate,
                                                        up_sampling=True))
            self.up_layer_name.append(f"decoder_{up_id}")
        self.final_layer = nn.Sequential(dropout_layer(conv_cfg, dropout_rate),
                                         build_conv_layer(conv_cfg,
                                                          in_channels=num_channels[0],
                                                          out_channels=extra.get('nclasses'),
                                                          kernel_size=1,
                                                          stride=1))
        if '2d' in conv_cfg['type']:
            self.interp_mode = 'bilinear'
        elif '3d' in conv_cfg['type']:
            self.interp_mode = 'trilinear'
        else:
            self.interp_mode = 'linear'

        # aux training layers length = len(down_block) - 2. The smallest feature map and the final layer do not need to do aux pred.
        if self.aux_train:
            self.aux_layers = nn.ModuleList()
            for nc in num_channels[1:-1][::-1]:
                self.aux_layers.append(nn.Sequential(dropout_layer(conv_cfg, dropout_rate),
                                                     build_conv_layer(conv_cfg,
                                                                      in_channels=nc,
                                                                      out_channels=extra.get('nclasses'),
                                                                      kernel_size=1,
                                                                      stride=1)))

    def forward(self, x):
        """Forward function."""
        origin_shape = x.shape
        encoder_outputs = []
        # encoding
        for stage_id, encoder_name in enumerate(self.down_layer_name):
            encoder_layer = getattr(self, encoder_name)
            x = encoder_layer(x)
            encoder_outputs.append(x)
        # decoding
        last_output = encoder_outputs[-1]
        if self.aux_train:
            aux_outputs = []
        for upstage_id, decoder_name in enumerate(self.up_layer_name[::-1]):
            decoder_layer = getattr(self, decoder_name)
            last_output = decoder_layer(last_output, encoder_outputs[-upstage_id - 2])
            if self.aux_train:
                if upstage_id < len(self.up_layer_name) - 1:
                    aux_outputs.append(F.interpolate(self.aux_layers[upstage_id](last_output), size=origin_shape[2:],
                                                     mode=self.interp_mode))
        last_output = self.final_layer(last_output)
        if last_output.shape[2:] != origin_shape[2:]:
            # linear interpolate
            last_output = F.interpolate(last_output, origin_shape[2:], mode=self.interp_mode)
        if self.aux_train:
            # in final out, decoder 3 out, decoder 2 out ... order
            return [last_output] + aux_outputs[::-1]
        else:
            return last_output

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
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


def _build_resunet(pretrained_name, depth, dimension="3d", in_chans=1, num_classes=1000, **kwargs):
    dimension = dimension.lower()
    assert dimension in ('1d', '2d', '3d'), "dimension must be 1d 2d or 3d"
    resnet_channels = {"10": (64, 64, 128, 256, 512),
                       "18": (64, 64, 128, 256, 512),
                       "34": (64, 64, 128, 256, 512),
                       "50": (64, 256, 512, 1024, 2048),
                       "101": (64, 256, 512, 1024, 2048),
                       "152": (64, 256, 512, 1024, 2048)}
    extra = dict(
        num_channels=resnet_channels[str(depth)],
        nclasses=num_classes,
    )
    conv_cfg = dict(type=f'Conv{dimension}')
    norm_cfg = dict(type=f'BN{dimension}', requires_grad=True)
    backbone_resnet = ResNet(depth, in_channels=in_chans, deep_stem=True,
                             first_stride=kwargs.get('first_stride', 2), conv_cfg=conv_cfg,
                             norm_cfg=norm_cfg)
    # unet
    unet_layers = [backbone_resnet.stem, nn.Sequential(backbone_resnet.maxpool,
                                                       getattr(backbone_resnet, backbone_resnet.res_layers[0]))] + [
                      getattr(backbone_resnet, x) for x in backbone_resnet.res_layers[1:]]

    # direct invoke
    # model = UNet(extra, unet_layers, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
    # return model

    # use cfg to build
    pretrained = False if pretrained_name is False or pretrained_name is None else True
    return build_model_with_cfg(model_cls=UNet, variant='unet', pretrained=pretrained,
                                default_cfg=default_cfgs.get(pretrained_name, None),
                                # model config
                                in_chans=in_chans, extra=extra, layers=unet_layers, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, **kwargs)


@register_model
def resunet50_3d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    depth = 50
    model = _build_resunet(pretrained, depth, dimension='3d', **kwargs)
    return model


@register_model
def resunet50_2d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    if pretrained is True:
        pretrained = 'resunet50_2d\tmultiplesclerosis'
    elif isinstance(pretrained, str):
        pretrained = 'resunet50_2d' + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger = logging.getLogger(__name__)
            _logger.warning(f"There is no {pretrained} pretrained weights, use random initialization.")
            pretrained = False

    depth = 50
    model = _build_resunet(pretrained, depth, dimension='2d', **kwargs)
    return model


@register_model
def resunet50_1d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    depth = 50
    model = _build_resunet(pretrained, depth, dimension='1d', **kwargs)
    return model
