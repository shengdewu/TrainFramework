import torch
import torch.nn as nn
import torchvision.models.vgg as tmv
from engine.loss.build import LOSS_ARCH_REGISTRY

__all__ = [
    'VggLoss'
]

VGG_ARCH_MAP = dict(
    vgg11='A',
    vgg13='B',
    vgg16='D',
    vgg19='E',
    vgg11_bn='A',
    vgg13_bn='B',
    vgg16_bn='D',
    vgg19_bn='E'
)


class VGG(tmv.VGG):
    def __init__(self, vgg_path: str, vgg_arch: str):
        assert vgg_arch in VGG_ARCH_MAP.keys(), f'not found vgg arch {vgg_arch} in {list(VGG_ARCH_MAP.keys())}'
        is_batch_norm = False if -1 == vgg_arch.find('bn') else True
        super(VGG, self).__init__(tmv.make_layers(tmv.cfgs[VGG_ARCH_MAP[vgg_arch]], batch_norm=is_batch_norm))
        state_dict = torch.load(vgg_path, map_location='cpu')
        self.load_state_dict(state_dict)
        del self.classifier
        del self.avgpool

        for param in self.features.parameters():
            param.requires_grad = False

        self.eval()
        return

    def forward(self, x):
        return self.features(x)


@LOSS_ARCH_REGISTRY.register()
class VggLoss(nn.Module):
    def __init__(self, vgg_path, vgg_arch: str, lambda_weight=1.):
        super(VggLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.vgg_op = VGG(vgg_path=vgg_path, vgg_arch=vgg_arch).to('cpu')
        self.vgg_path = vgg_path
        self.lambda_weight = lambda_weight
        self.vgg_arch = vgg_arch
        self.citerion_l2 = nn.MSELoss()
        return

    def forward(self, x, target):
        """
        :param x: b, c, h, w
        :param target: b, c, h, w
        :return:
        """
        assert x.shape == target.shape
        real = self.vgg_op(x)
        fake = self.vgg_op(target)
        loss = self.citerion_l2(real, fake)
        return self.lambda_weight * loss

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'vgg_path: {}, '.format(self.vgg_path)
        format_string += 'vgg_arch: {}'.format(self.vgg_arch)
        format_string += 'lambda_weight: {})'.format(self.lambda_weight)
        return format_string

