import torch
from .build import LOSS_ARCH_REGISTRY

__all__ = [
    'MSELoss'
]


@LOSS_ARCH_REGISTRY.register()
class MSELoss(torch.nn.Module):
    def __init__(self, lambda_weight=1.):
        super(MSELoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.l2_op = torch.nn.MSELoss()
        return

    def forward(self, x, target):
        """
        :param x: b, c, h, w
        :param target: b, c, h, w
        :return:
        """
        assert x.shape == target.shape
        loss = self.l2_op(x, target)
        return self.lambda_weight * loss

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'lambda_weight: {})'.format(self.lambda_weight)
        return format_string

