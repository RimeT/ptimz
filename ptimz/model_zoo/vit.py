# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import MultiheadAttention, to_ntuple
from mmcls.utils import get_root_logger
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList

from .helpers import build_model_with_cfg
from .registry import register_model
from .vision_transformer.embed import resize_pos_embed, chan_first_reord
from .vision_transformer.transformer import FFN, PatchEmbed

__all__ = ['VisionTransformer']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        # 'interpolation': 'trilinear',
        'num_classes': 1000,
        **kwargs
    }


default_cfgs = {
}


def check_interpolate_mode(mode, dimension):
    if 'cubic' in mode:
        if 2 == dimension:
            return 'bicubic'
        if 3 == dimension:
            return 'trilinear'
        if 1 == dimension:
            return 'linear'
    if 'linear' in mode:
        if 2 == dimension:
            return 'bilinear'
        if 3 == dimension:
            return 'trilinear'
        if 1 == dimension:
            return 'linear'
    return mode


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x


class VisionTransformer(BaseBackbone):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    # Some structures have multiple extra tokens, like DeiT.
    num_extra_tokens = 1  # cls_token

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None,
                 head_type=None,
                 num_classes=1000,
                 **kwargs
                 ):
        super(VisionTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)

        conv_type = _patch_cfg.get('conv_type').lower()
        if '2d' in conv_type:
            self.dimension = 2
        elif '3d' in conv_type:
            self.dimension = 3
        elif '1d' in conv_type:
            self.dimension = 1
        else:
            raise ValueError(f'Dimension {conv_type} not support.')

        self.img_size = to_ntuple(self.dimension)(img_size)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        num_patches = self.patch_resolution[0]
        for pr in self.patch_resolution[1:]:
            num_patches *= pr

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                                           f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # Set position embedding
        self.interpolate_mode = check_interpolate_mode(interpolate_mode, self.dimension)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                    arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.head_type = head_type
        if 'classification' == head_type:
            self.head_fc = nn.Linear(self.embed_dims, num_classes)
        else:
            self.head_fc = None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(VisionTransformer, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmcv.utils import print_log
            logger = get_root_logger()
            print_log(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.',
                logger=logger)

            ckpt_pos_embed_shape = to_ntuple(self.dimension)(
                # the original mmcls used int(), for some pow(1/3), round value is more close
                round(np.power(ckpt_pos_embed_shape[1] - self.num_extra_tokens, 1 / self.dimension)))

            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(*chan_first_reord(self.dimension))
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(*chan_first_reord(self.dimension))
                    cls_token = None
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

        if 'classification' == self.head_type and self.output_cls_token:
            outs = outs[-1]
            _, cls_token = outs
            return self.head_fc(cls_token)

        return tuple(outs)


def _build_vit(pretrained_name, arch, dimension, patch_size, image_size, in_chans=1, num_classes=1000, head_type=None,
               **kwargs):
    dimension = dimension.lower()
    assert dimension in ('1d', '2d', '3d'), "dimension must be 1d 2d or 3d"
    patch_cfg = dict(conv_type=f'Conv{dimension}')

    # use cfg to build
    pretrained = False if pretrained_name is False or pretrained_name is None else True
    return build_model_with_cfg(model_cls=VisionTransformer, variant='vit-base', pretrained=pretrained,
                                default_cfg=default_cfgs.get(pretrained_name, None),
                                # model config
                                arch=arch,
                                patch_size=patch_size,
                                image_size=image_size,
                                patch_cfg=patch_cfg,
                                in_channels=in_chans,
                                head_type=head_type,
                                num_classes=num_classes,
                                **kwargs)


@register_model
def vit_base_patch16_224_cls2d(pretrained=False, **kwargs):
    _logger = logging.getLogger(__name__)
    func_name = str(inspect.stack()[0][3])
    if pretrained is True:
        _logger.warning(f"{func_name} has no pretrained weights, use random initialization.")
        pretrained = False
    elif isinstance(pretrained, str):
        pretrained = func_name + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger.warning(f"{func_name} no {pretrained} pretrained weights, use random initialization.")
            pretrained = False
    return _build_vit(pretrained, arch='base', dimension='2d', patch_size=16, image_size=224,
                      head_type='classification', **kwargs)


@register_model
def vit_base_patch16_224_cls3d(pretrained=False, **kwargs):
    _logger = logging.getLogger(__name__)
    func_name = str(inspect.stack()[0][3])
    if pretrained is True:
        _logger.warning(f"{func_name} has no pretrained weights, use random initialization.")
        pretrained = False
    elif isinstance(pretrained, str):
        pretrained = func_name + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger.warning(f"{func_name} no {pretrained} pretrained weights, use random initialization.")
            pretrained = False
    return _build_vit(pretrained, arch='base', dimension='3d', patch_size=16, image_size=224,
                      head_type='classification', **kwargs)


@register_model
def vit_base_cls3d(patch_size, image_size, pretrained=False, **kwargs):
    """

    :param patch_size: image embedding size
    :param image_size: coarse image size, e.g. image_size=224. This param is to get vit sequence_length for pos_embed
    :param pretrained:
    :param kwargs:
    :return:
    """
    _logger = logging.getLogger(__name__)
    func_name = str(inspect.stack()[0][3])
    if pretrained is True:
        _logger.warning(f"{func_name} has no pretrained weights, use random initialization.")
        pretrained = False
    elif isinstance(pretrained, str):
        pretrained = func_name + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger.warning(f"{func_name} no {pretrained} pretrained weights, use random initialization.")
            pretrained = False
    return _build_vit(pretrained, arch='base', dimension='3d', patch_size=patch_size, image_size=image_size,
                      head_type='classification', **kwargs)


@register_model
def vit_base_cls2d(patch_size, image_size, pretrained=False, **kwargs):
    """

    :param patch_size: image embedding size
    :param image_size: coarse image size, e.g. image_size=224. This param is to get vit sequence_length for pos_embed
    :param pretrained:
    :param kwargs:
    :return:
    """
    _logger = logging.getLogger(__name__)
    func_name = str(inspect.stack()[0][3])
    if pretrained is True:
        _logger.warning(f"{func_name} has no pretrained weights, use random initialization.")
        pretrained = False
    elif isinstance(pretrained, str):
        pretrained = func_name + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger.warning(f"{func_name} no {pretrained} pretrained weights, use random initialization.")
            pretrained = False
    return _build_vit(pretrained, arch='base', dimension='2d', patch_size=patch_size, image_size=image_size,
                      head_type='classification', **kwargs)