from typing import Optional

import torch


def _channelwise_sum(x: torch.Tensor):
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    reduce_dims = tuple([0] + list(range(x.dim()))[2:])  # = (0, 2, 3, ...)
    return x.sum(dim=reduce_dims)


def dice_loss(probs, target, weight=1.0, eps=0.0001, smooth=0.0, cls_weight=None):
    tsh, psh = target.shape, probs.shape

    if tsh == psh:  # Already one-hot
        onehot_target = target.to(probs.dtype)
    elif (
            tsh[0] == psh[0] and tsh[1:] == psh[2:]
    ):  # Assume dense target storage, convert to one-hot
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(
            f"Target shape {target.shape} is not compatible with output shape {probs.shape}."
        )

    intersection = probs * onehot_target  # (N, C, ...)
    numerator = 2 * _channelwise_sum(intersection) + smooth  # (C,)
    denominator = probs + onehot_target  # (N, C, ...)
    denominator = _channelwise_sum(denominator) + smooth + eps  # (C,)
    loss_per_channel = 1 - (numerator / denominator)  # (C,)
    weighted_loss_per_channel = weight * loss_per_channel  # (C,)
    if cls_weight is not None:
        weighted_loss_per_channel = cls_weight.to(weighted_loss_per_channel.device) * weighted_loss_per_channel
    return weighted_loss_per_channel.mean()  # ()


class DiceLoss(torch.nn.Module):
    def __init__(
            self,
            apply_softmax: bool = True,
            weight: Optional[torch.Tensor] = None,
            cls_weight: Optional[torch.Tensor] = None,
            smooth: float = 0.0,
    ):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = dice_loss
        if weight is None:
            weight = torch.tensor(1.0)
        self.register_buffer("weight", weight)
        self.smooth = smooth
        self.cls_weight = cls_weight

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight, smooth=self.smooth, cls_weight=self.cls_weight)
