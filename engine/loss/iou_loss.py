import torch
import torch.nn.functional as F
from engine.loss.build import LOSS_ARCH_REGISTRY


__all__ = [
    'IOULoss'
]


@LOSS_ARCH_REGISTRY.register()
class IOULoss(torch.nn.Module):
    def __init__(self, lambda_weight=1.,
                 apply_sigmoid: bool = True,
                 ignore_index: int = None
                 ):
        super(IOULoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.apply_sigmoid = apply_sigmoid
        self.ignore_index = ignore_index
        return

    def forward(self, predict, target):
        """
        :param predict: b, c, h, w and c is class number
        :param target: b, 1, h, w and 0 < targets[i] < c
        :return:
        """
        if self.apply_sigmoid:
            predict = F.sigmoid(predict)
        else:
            predict = torch.softmax(predict, dim=1)
            # target to one hot format
        if target.size() == predict.size():
            labels_one_hot = target
        elif target.dim() == 3:  # if target tensor is in class indexes format.
            if predict.size(1) == 1 and self.ignore_index is None:  # if one class prediction task
                labels_one_hot = target.unsqueeze(1)
            else:
                labels_one_hot = F.one_hot(
                    torch.clamp(target.long(), 0, predict.shape[1] - 1),
                    num_classes=predict.shape[1]).permute(0, 3, 1, 2)
        elif target.dim() == 4 and target.shape[1] == 1:
            target = target[:, 0, :, :]
            if predict.size(1) == 1 and self.ignore_index is None:  # if one class prediction task
                labels_one_hot = target.unsqueeze(1)
            else:
                labels_one_hot = F.one_hot(
                    torch.clamp(target.long(), 0, predict.shape[1] - 1),
                    num_classes=predict.shape[1]).permute(0, 3, 1, 2)
        else:
            raise AssertionError(
                f"Mismatch of target shape: {target.size()} and prediction shape: {predict.size()},"
                f" target must be [NxWxH] tensor for to_one_hot conversion"
                f" or to have the same num of channels like prediction tensor"
            )

        if self.ignore_index is not None:
            if target.shape != predict.shape:
                # target N H W
                valid_mask = target.ne(self.ignore_index).unsqueeze(1).expand_as(predict)
            else:
                # target H 1 H W
                valid_mask = target.ne(self.ignore_index).expand_as(predict)
            predict = predict * valid_mask
            labels_one_hot = labels_one_hot * valid_mask

        iou_and = torch.sum(predict * labels_one_hot)
        iou_or = torch.sum(predict) + torch.sum(labels_one_hot) - iou_and
        iou = iou_and / iou_or

        return self.lambda_weight * (iou + (1 - iou))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ' ,lambda_weight: {}, '.format(self.lambda_weight)
        format_string += ' ,apply_sigmoid: {}, '.format(self.apply_sigmoid)
        format_string += ' ,ignore_index: {})'.format(self.ignore_index)
        return format_string
