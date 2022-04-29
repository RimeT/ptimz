from typing import Optional, Sequence

import torch


class AuxLoss(torch.nn.Module):

    def __init__(self,
                 criteria: torch.nn.Module,
                 weight: Optional[Sequence[float]] = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.criteria = criteria
        self.device = device
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight.to(self.device))

    def forward(self, outputs, label):
        loss = torch.tensor(0.0, device=outputs[0].device)
        if self.weight is None:
            self.weight = torch.ones((len(outputs),))
        for o, w in zip(outputs, self.weight):
            l = self.criteria(o, label) * w
            loss += l
        return loss


class CombinedLoss(torch.nn.Module):
    """Defines a loss function as a weighted sum of combinable loss criteria.
    Args:
        criteria: List of loss criterion modules that should be combined.
        weight: Weight assigned to the individual loss criteria (in the same
            order as ``criteria``).
        device: The device on which the loss should be computed. This needs
            to be set to the device that the loss arguments are allocated on.
    """

    def __init__(
            self,
            criteria: Sequence[torch.nn.Module],
            weight: Optional[Sequence[float]] = None,
            device: torch.device = None,
    ):
        super().__init__()
        self.criteria = torch.nn.ModuleList(criteria)
        self.device = device
        if weight is None:
            weight = torch.ones(len(criteria))
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            assert weight.shape == (len(criteria),)
        self.register_buffer("weight", weight.to(self.device))

    def forward(self, *args):
        loss = torch.tensor(0.0, device=args[0].device)
        for crit, weight in zip(self.criteria, self.weight):
            l = crit(*args)
            loss += weight * crit(*args)
        return loss
