import torch
from .function.ssim import SSIM
from .build import LOSS_ARCH_REGISTRY

__all__ = [
    'SSIMLoss'
]


@LOSS_ARCH_REGISTRY.register()
class SSIMLoss(torch.nn.Module):
    def __init__(self, lambda_weight=1.):
        super(SSIMLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.ssim_op = SSIM()
        return

    def forward(self, x, target):
        """
        :param x: b, c, h, w
        :param target: b, C, h, w
        :return:
        """
        assert x.shape == target.shape
        loss = 1.0 - self.ssim_op(x, target)
        return self.lambda_weight * loss

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'lambda_weight: {})'.format(self.lambda_weight)
        return format_string

