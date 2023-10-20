import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
from .build import LOSS_ARCH_REGISTRY

__all__ = [
    'L1Loss',
    'NLLLoss',
    'MSELoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'SmoothL1Loss',
    'SoftMarginLoss',
    'CrossEntropyLoss',
    'MultiLabelSoftMarginLoss',
    'TripletMarginLoss',
    'TripletMarginWithDistanceLoss'
]


@LOSS_ARCH_REGISTRY.register()
class L1Loss(nn.L1Loss):

    def __init__(self, lambda_weight: float = 1., size_average=None, reduction: str = 'mean', ) -> None:
        super(L1Loss, self).__init__(size_average, reduction=reduction)
        self.lambda_weight = lambda_weight
        return

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(L1Loss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class NLLLoss(nn.NLLLoss):
    def __init__(self, lambda_weight: float = 1., weight: Optional[Tensor] = None, ignore_index: int = -100, reduction: str = 'mean') -> None:
        super(NLLLoss, self).__init__(weight, ignore_index=ignore_index, reduction=reduction)
        self.lambda_weight = lambda_weight

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(NLLLoss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class MSELoss(nn.MSELoss):
    def __init__(self, lambda_weight: float = 1., reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(reduction=reduction)
        self.lambda_weight = lambda_weight
        return

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(MSELoss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class BCELoss(nn.BCELoss):
    def __init__(self, lambda_weight: float = 1., weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super(BCELoss, self).__init__(weight=weight, reduction=reduction)
        self.lambda_weight = lambda_weight

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(BCELoss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, lambda_weight: float = 1., weight: Optional[Tensor] = None, reduction: str = 'mean', pos_weight: Optional[Tensor] = None) -> None:
        super(BCEWithLogitsLoss, self).__init__(weight, reduction=reduction, pos_weight=pos_weight)
        self.lambda_weight = lambda_weight

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(BCEWithLogitsLoss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class SmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, lambda_weight: float = 1., reduction: str = 'mean', beta: float = 1.0) -> None:
        super(SmoothL1Loss, self).__init__(reduction=reduction, beta=beta)
        self.lambda_weight = lambda_weight
        return

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(SmoothL1Loss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class SoftMarginLoss(nn.SoftMarginLoss):
    def __init__(self, lambda_weight: float = 1., reduction: str = 'mean') -> None:
        super(SoftMarginLoss, self).__init__(reduction=reduction)
        self.lambda_weight = lambda_weight

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(SoftMarginLoss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, lambda_weight: float = 1., weight: Optional[Tensor] = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(CrossEntropyLoss, self).__init__(weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
        self.lambda_weight = lambda_weight

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(CrossEntropyLoss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class MultiLabelSoftMarginLoss(nn.MultiLabelSoftMarginLoss):
    def __init__(self, lambda_weight: float = 1., weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super(MultiLabelSoftMarginLoss, self).__init__(weight, reduction=reduction)
        self.lambda_weight = lambda_weight

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return super(MultiLabelSoftMarginLoss, self).forward(input_tensor, target_tensor) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class TripletMarginLoss(nn.TripletMarginLoss):
    def __init__(self, lambda_weight: float = 1., margin: float = 1.0, p: float = 2., eps: float = 1e-6, swap: bool = False, reduction: str = 'mean'):
        super(TripletMarginLoss, self).__init__(margin, p=p, eps=eps, swap=swap, reduction=reduction)
        self.lambda_weight = lambda_weight

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return super(TripletMarginLoss, self).forward(anchor, positive, negative) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class TripletMarginWithDistanceLoss(nn.TripletMarginWithDistanceLoss):
    def __init__(self, lambda_weight: float = 1., distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
                 margin: float = 1.0, swap: bool = False, reduction: str = 'mean'):
        super(TripletMarginWithDistanceLoss, self).__init__(distance_function=distance_function, margin=margin, swap=swap, reduction=reduction)
        self.lambda_weight = lambda_weight

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return super(TripletMarginWithDistanceLoss, self).forward(anchor, positive, negative) * self.lambda_weight
