from engine.loss.build import LOSS_ARCH_REGISTRY
from typing import Union, Tuple
from enum import Enum
import torch
import torch.nn.functional as F

__all__ = [
    'BinaryDiceLoss',
    'GeneralizedDiceLoss'
]


class _LossReduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"


class _DiceLossBase(torch.nn.Module):
    """
    Compute average Dice loss between two tensors, It can support both multi-classes and binary tasks.
    Defined in the paper: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    """

    def __init__(
            self,
            apply_softmax: bool = True,
            ignore_index: int = None,
            smooth: float = 1.0,
            eps: float = 1e-5,
            reduce_over_batches: bool = False,
            generalized_metric: bool = False,
            weight: Union[float, torch.Tensor] = 1.0,
            reduction: Union[_LossReduction, str] = "mean",
            lambda_weight: float = 1.
    ):
        """
        :param apply_softmax: Whether to apply softmax to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the metric
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        :param reduce_over_batches: Whether to average metric over the batch axis if set True,
         default is `False` to average over the classes axis.
        :param generalized_metric: Whether to apply normalization by the volume of each class.
        :param weight: a manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`.
        :param reduction: Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
            `none`: no reduction will be applied.
            `mean`: the sum of the output will be divided by the number of elements in the output.
            `sum`: the output will be summed.
            Default: `mean`
        :param lambda_weight
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.apply_softmax = apply_softmax
        self.eps = eps
        self.smooth = smooth
        self.reduce_over_batches = reduce_over_batches
        self.generalized_metric = generalized_metric
        self.weight = weight
        self.lambda_weight = lambda_weight
        if self.generalized_metric:
            assert self.weight is None, "Cannot use structured Loss with weight classes and generalized normalization"
            if self.eps > 1e-12:
                print("warning When using GeneralizedLoss, it is recommended to use eps below 1e-12, to not affect" "small values normalized terms.")
            if self.smooth != 0:
                print("warning When using GeneralizedLoss, it is recommended to set smooth value as 0.")
        self.reduction = reduction
        return

    def _calc_numerator_denominator(self, labels_one_hot: torch.tensor, predict: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Calculate dice metric's numerator and denominator.

        :param labels_one_hot: target in one hot format.   shape: [BS, num_classes, img_width, img_height]
        :param predict: predictions tensor.                shape: [BS, num_classes, img_width, img_height]
        :return:
            numerator = intersection between predictions and target. shape: [BS, num_classes, img_width, img_height]
            denominator = sum of predictions and target areas.       shape: [BS, num_classes, img_width, img_height]
        """
        numerator = labels_one_hot * predict
        denominator = labels_one_hot + predict
        return numerator, denominator

    def _calc_loss(self, numerator: torch.tensor, denominator: torch.tensor) -> torch.tensor:
        """
        Calculate dice loss.
        All tensors are of shape [BS] if self.reduce_over_batches else [num_classes].

        :param numerator: intersection between predictions and target.
        :param denominator: total number of pixels of prediction and target.
        """
        loss = 1.0 - ((2.0 * numerator + self.smooth) / (denominator + self.eps + self.smooth))
        return loss

    def apply_reduce(self, loss: torch.Tensor):
        if self.reduction == _LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == _LossReduction.SUM.value:
            loss = loss.sum()
        elif not _LossReduction.NONE.value:
            raise ValueError(f"Reduction mode is not supported, expected options are ['mean', 'sum', 'none']" f", found {self.reduction}")
        return loss

    def to_one_hot(self, target: torch.Tensor, num_classes: int):
        """
        Target label to one_hot tensor. labels and ignore_index must be consecutive numbers.
        :param target: Class labels long tensor, with shape [N, H, W]
        :param num_classes: num of classes in datasets excluding ignore label, this is the output channels of the one hot
            result.
        :return: one hot tensor with shape [N, num_classes, H, W]
        """
        num_classes = num_classes if self.ignore_index is None else num_classes + 1

        one_hot = F.one_hot(target, num_classes).permute((0, 3, 1, 2))

        if self.ignore_index is not None:
            # remove ignore_index channel
            one_hot = torch.cat([one_hot[:, :self.ignore_index], one_hot[:, self.ignore_index + 1:]], dim=1)

        return one_hot

    def forward(self, predict, target):
        if self.apply_softmax:
            predict = torch.softmax(predict, dim=1)
        # target to one hot format
        if target.size() == predict.size():
            labels_one_hot = target
        elif target.dim() == 3:  # if target tensor is in class indexes format.
            if predict.size(1) == 1 and self.ignore_index is None:  # if one class prediction task
                labels_one_hot = target.unsqueeze(1)
            else:
                labels_one_hot = self.to_one_hot(target, num_classes=predict.shape[1])
        else:
            raise AssertionError(
                f"Mismatch of target shape: {target.size()} and prediction shape: {predict.size()},"
                f" target must be [NxWxH] tensor for to_one_hot conversion"
                f" or to have the same num of channels like prediction tensor"
            )

        reduce_spatial_dims = list(range(2, len(predict.shape)))
        reduce_dims = [1] + reduce_spatial_dims if self.reduce_over_batches else [0] + reduce_spatial_dims

        # Calculate the numerator and denominator of the chosen metric
        numerator, denominator = self._calc_numerator_denominator(labels_one_hot, predict)

        # exclude ignore labels from numerator and denominator, false positive predicted on ignore samples
        # are not included in the total calculation.
        if self.ignore_index is not None:
            valid_mask = target.ne(self.ignore_index).unsqueeze(1).expand_as(denominator)
            numerator *= valid_mask
            denominator *= valid_mask

        numerator = torch.sum(numerator, dim=reduce_dims)
        denominator = torch.sum(denominator, dim=reduce_dims)

        if self.generalized_metric:
            weights = 1.0 / (torch.sum(labels_one_hot, dim=reduce_dims) ** 2)
            # if some classes are not in batch, weights will be inf.
            infs = torch.isinf(weights)
            weights[infs] = 0.0
            numerator *= weights
            denominator *= weights

        # Calculate the loss of the chosen metric
        losses = self._calc_loss(numerator, denominator)
        if self.weight is not None:
            losses *= self.weight
        return self.apply_reduce(losses) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class BinaryDiceLoss(_DiceLossBase):
    """
    Compute Dice Loss for binary class tasks (1 class only).
    Except target to be a binary map with 0 and 1 values.
    """

    def __init__(self, apply_sigmoid: bool = True, smooth: float = 1.0, eps: float = 1e-5, weight: Union[float, torch.Tensor] = None, lambda_weight: float = 1.):
        """
        :param apply_sigmoid: Whether to apply sigmoid to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the dice
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        """
        super().__init__(apply_softmax=False, ignore_index=None, smooth=smooth, eps=eps, reduce_over_batches=False, weight=weight, lambda_weight=lambda_weight)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, predict: torch.tensor, target: torch.tensor) -> torch.tensor:
        if self.apply_sigmoid:
            predict = torch.sigmoid(predict)
        return super().forward(predict=predict, target=target)


@LOSS_ARCH_REGISTRY.register()
class GeneralizedDiceLoss(_DiceLossBase):
    """
    Compute the Generalised Dice loss, contribution of each label is normalized by the inverse of its volume, in order
     to deal with class imbalance.
    Defined in the paper: "Generalised Dice overlap as a deep learning loss function for highly unbalanced
     segmentations"

    :param smooth:  default value is 0, smooth laplacian is not recommended to be used with GeneralizedDiceLoss.
         because the weighted values to be added are very small.
    :param eps:     default value is 1e-17, must be a very small value, because weighted `intersection` and
        `denominator` are very small after multiplication with `1 / counts ** 2`
    """

    def __init__(
            self,
            apply_softmax: bool = True,
            ignore_index: int = None,
            smooth: float = 0.0,
            eps: float = 1e-17,
            reduce_over_batches: bool = False,
            reduction: Union[_LossReduction, str] = "mean",
            weight: Union[float, torch.Tensor] = None,
            lambda_weight: float = 1.0
    ):
        """
        :param apply_softmax: Whether to apply softmax to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the dice
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        :param reduce_over_batches: Whether to apply reduction over the batch axis if set True,
         default is `False` to average over the classes axis.
        :param reduction: Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
            `none`: no reduction will be applied.
            `mean`: the sum of the output will be divided by the number of elements in the output.
            `sum`: the output will be summed.
            Default: `mean`
        """
        super().__init__(
            apply_softmax=apply_softmax,
            ignore_index=ignore_index,
            smooth=smooth,
            eps=eps,
            reduce_over_batches=reduce_over_batches,
            generalized_metric=True,
            weight=weight,
            reduction=reduction,
            lambda_weight=lambda_weight
        )
