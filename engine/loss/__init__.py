from engine.loss.ssim_loss import SSIMLoss
from engine.loss.torch_loss import *
from engine.loss.vgg_loss import *
from engine.loss.dice_loss import *
from engine.loss.iou_loss import *
from .pipe import LossKeyCompose, LossCompose
from .build import LOSS_ARCH_REGISTRY
