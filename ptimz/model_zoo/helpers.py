""" Model creation / weight loading / state_dict helpers
"""
import logging
from copy import deepcopy
from typing import Any, Callable, Optional

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

_logger = logging.getLogger(__name__)


def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def adapt_input_conv(in_chans, conv_weight):
    # use linear interpolate on channels
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    weight_shape = list(conv_weight.shape)
    conv_weight = conv_weight.permute(0, *list(range(2, len(weight_shape))), 1)
    conv_weight = conv_weight.reshape(-1, 1, weight_shape[1])
    conv_weight = F.interpolate(conv_weight, in_chans, mode='linear')
    conv_weight = conv_weight.reshape(weight_shape[0], *weight_shape[2:], in_chans).permute(0, len(weight_shape) - 1,
                                                                                            *list(range(1,
                                                                                                        len(weight_shape) - 1)))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained(model, default_cfg=None, num_classes=2, in_chans=1, filter_fn=None, strict=True, progress=False):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        progress (bool): enable progress bar for weight download

    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    hf_hub_id = default_cfg.get('hf_hub', None)
    if not pretrained_url and not hf_hub_id:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if pretrained_url:
        _logger.info(f'Loading pretrained weights from url ({pretrained_url})')
        state_dict = load_state_dict_from_url(pretrained_url, progress=progress, map_location='cpu')
    # elif hf_hub_id and has_hf_hub(necessary=True):
    #     _logger.info(f'Loading pretrained weights from Hugging Face hub ({hf_hub_id})')
    #     state_dict = load_state_dict_from_hf(hf_hub_id)
    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    # ptimz pretrained pth is a dict include state_dict as model weights
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    input_convs = default_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != default_cfg.get('input_size')[0]:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from {default_cfg.get("in_channels")} to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    last_layers = default_cfg.get('last_layer', None)
    if last_layers is not None:
        if isinstance(last_layers, str):
            last_layers = (last_layers,)
        if num_classes != default_cfg.get('num_classes', 2):
            for last_layer_name in last_layers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                del state_dict[last_layer_name + '.weight']
                del state_dict[last_layer_name + '.bias']
                strict = False

    model.load_state_dict(state_dict, strict=strict)


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: Optional[Any] = None,
        # feature_cfg: Optional[dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        **kwargs):
    pruned = kwargs.pop('pruned', False)
    default_cfg = deepcopy(default_cfg) if default_cfg else {}

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg

    # if pruned:
    #     model = adapt_model_from_file(model, variant)

    if 'in_channels' in kwargs:
        in_chans = kwargs['in_channels']
    elif 'in_channel' in kwargs:
        in_chans = kwargs['in_channel']
    elif 'in_chans' in kwargs:
        in_chans = kwargs['in_chans']
    else:
        raise NotImplementedError(f"in_chans not found in model parameters.")

    if pretrained:
        # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
        num_classes_pretrained = getattr(model, 'num_classes', kwargs.get('num_classes', 0))
        load_pretrained(
            model,
            num_classes=num_classes_pretrained,
            in_chans=in_chans,
            filter_fn=pretrained_filter_fn,
            strict=pretrained_strict)

    return model
