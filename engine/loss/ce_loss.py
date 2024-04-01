import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from .build import LOSS_ARCH_REGISTRY

from .util import LossReduction

__all__ = [
    'GeneralizedCELoss'
]


@LOSS_ARCH_REGISTRY.register()
class GeneralizedCELoss(nn.Module):
    def __init__(self, apply_sigmoid: bool = False,
                 ignore_index: int = None,
                 reduction: str = "mean",
                 lambda_weight: float = 1.
                 ):
        super(GeneralizedCELoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lambda_weight = lambda_weight
        self.apply_sigmoid = apply_sigmoid
        return

    def forward(self, predict, target):
        if target.dim() == 4 and target.shape[1] == 1:  # if target tensor is in class indexes format.
            target = target[:, 0, :, :]
        elif target.dim() != 3:
            raise AssertionError(
                f"Mismatch of target shape: {target.size()} and prediction shape: {predict.size()},"
                f" target must be [NxWxH] or [Nx1xWxH tensor"
            )

        if self.apply_sigmoid:
            loss = self.binary_cross_entropy(predict, target)
        else:
            loss = self.cross_entropy(predict, target)
        return loss * self.lambda_weight

    def cross_entropy(self, pred, label):
        """cross_entropy. The wrapper function for :func:`F.cross_entropy`

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
        """
        loss = F.cross_entropy(
            pred,
            label,
            reduction='none',
            ignore_index=self.ignore_index)

        return self.apply_reduce(loss)

    def binary_cross_entropy(self, pred, label):
        """Calculate the binary CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
                Note: In bce loss, label < 0 is invalid.
        Returns:
            torch.Tensor: The calculated loss
        """
        if pred.size(1) == 1:
            # For binary class segmentation, the shape of pred is
            # [N, 1, H, W] and that of label is [N, H, W].
            # As the ignore_index often set as 255, so the
            # binary class label check should mask out
            # ignore_index
            assert label[label != self.ignore_index].max() <= 1, \
                'For pred with shape [N, 1, H, W], its label must have at ' \
                'most 2 classes'
            pred = F.sigmoid(pred.squeeze(1))

        if pred.dim() != label.dim():
            assert (pred.dim() == 2 and label.dim() == 1) or (
                    pred.dim() == 4 and label.dim() == 3), \
                'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
                'H, W], label shape [N, H, W] are supported'
            # has been treated for valid (non-ignore) pixels
            label, valid_mask = self._expand_onehot_labels(label, pred.shape)
        else:
            # should mask out the ignored elements
            valid_mask = ((label >= 0) & (label != self.ignore_index)).float()

        loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction='none')
        # do the reduction for the weighted loss
        return self.apply_reduce(loss * valid_mask)

    def _expand_onehot_labels(self, labels, target_shape):
        """Expand onehot labels to match the size of prediction."""
        bin_labels = labels.new_zeros(target_shape)
        valid_mask = (labels >= 0) & (labels != self.ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)

        if inds[0].numel() > 0:
            if labels.dim() == 3:
                bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
            else:
                bin_labels[inds[0], labels[valid_mask]] = 1

        valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()

        return bin_labels, valid_mask

    def apply_reduce(self, loss: Tensor):
        if self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.SUM.value:
            loss = loss.sum()
        elif not LossReduction.NONE.value:
            raise ValueError(f"Reduction mode is not supported, expected options are ['mean', 'sum', 'none']" f", found {self.reduction}")
        return loss

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ' ,reduction: {}, '.format(self.reduction)
        format_string += ' ,lambda_weight: {}, '.format(self.lambda_weight)
        format_string += ' ,apply_sigmoid: {}, '.format(self.apply_sigmoid)
        format_string += ' ,ignore_index: {})'.format(self.ignore_index)
        return format_string
