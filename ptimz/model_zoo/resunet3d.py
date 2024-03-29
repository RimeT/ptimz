
# -*- coding: utf-8 -*-
"""
#reference https://github.com/assassint2017/MICCAI-LITS2017/blob/master/README.md
"""


import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import build_model_with_cfg
from .registry import register_model

__all__ = ['ResUNet_3d']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        # 'interpolation': 'trilinear',
        'num_classes': 1000,
        **kwargs
    }


default_cfgs = {
    "resunet_3d\tctliver": _cfg(
        url='https://github.com/songphilips/ptimz/releases/download/v0.0.1-hrnet/resunet-3d-net990-0.013-0.018.pth',
        input_details='CT [lits17]',
        slice_thickness=1,
        first_conv='encoder_stage1.0',
        num_classes=1, input_size=(1, 256, 256), last_layer=('map1.0','map2.0','map3.0','map4.0')),
}

def _build_resunet_3d(pretrained_name,  in_chans=1, num_classes=1,  **kwargs):
    pretrained = False if pretrained_name is False or pretrained_name is None else True
    if 'training' in kwargs.keys():
        train_act = kwargs.get('training')
    else:
        train_act = False
    return build_model_with_cfg(ResUNet_3d, variant='resunet3d', pretrained=pretrained,
                                default_cfg=default_cfgs.get(pretrained_name, None),
                                # model config
                                in_channels=in_chans,
                                num_classes=num_classes,
                                training = train_act)
def _check_pretrained(pretrained, netname='resunet_3d'):
    if pretrained is True:
        pretrained = False
    elif isinstance(pretrained, str):
        pretrained = netname + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger = logging.getLogger(__name__)
            _logger.warning(f"There is no {pretrained} pretrained weights, use random initialization.")
            pretrained = False
    return pretrained

class ResUNet_3d(nn.Module):

    def __init__(self, training=False,in_channels=1,num_classes=1):
        super().__init__()

        self.training = training
        self.in_channels = in_channels
        self.num_classes = num_classes #foreground classes, not include background class

        assert self.in_channels == 1, 'Input channel should be 1. Multi-channel input will be available in further update'

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(self.in_channels, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        
        self.map4 = nn.Sequential(
            nn.Conv3d(32, self.num_classes, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        
        self.map3 = nn.Sequential(
            nn.Conv3d(64, self.num_classes, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

       
        self.map2 = nn.Sequential(
            nn.Conv3d(128, self.num_classes, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        
        self.map1 = nn.Sequential(
            nn.Conv3d(256, self.num_classes, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, 0.3, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, 0.3, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, 0.3, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, 0.3, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)



@register_model
def resunet_3d(pretrained=False, **kwargs):
    pretrained = _check_pretrained(pretrained, f'resunet_3d')
    model = _build_resunet_3d(pretrained, **kwargs)
    if pretrained is False:
        model.apply(init)
    return model



