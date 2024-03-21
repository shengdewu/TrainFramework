import torch
from engine.loss.build import LOSS_ARCH_REGISTRY


__all__ = [
    'IOULoss'
]


def _iou(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


@LOSS_ARCH_REGISTRY.register()
class IOULoss(torch.nn.Module):
    def __init__(self, lambda_weight=1.):
        super(IOULoss, self).__init__()
        self.lambda_weight = lambda_weight
        return

    def forward(self, x, target):
        """
        :param x: b, c, h, w and c is class number
        :param target: b, 1, h, w and 0 < targets[i] < c
        :return:
        """
        assert len(x.shape) == len(target.shape) == 4 and target.shape[1] == 1
        if x.shape[1] > 1:
            x = torch.argmax(x, dim=1).unsqueeze(1)
        return self.lambda_weight * _iou(x, target)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ' ,lambda_weight: {})'.format(self.lambda_weight)
        return format_string
