from collections.abc import Sequence
import sys
import torch
import logging
import random
from torchvision.transforms import functional as ttf
import numpy as np


def range_float(start, end, step, exclude):
    assert start <= end
    if step >= 1 and int(step*10) == (step // 1)*10:
        return [i for i in np.arange(int(start), int(end+1), step=int(step)) if i != int(exclude)]
    else:
        base = pow(10, len(str(step).split('.')[1]))
        return [i/base for i in np.arange(int(start*base), int(end*base+1), step=int(step*base)) if i != int(exclude*base)]


class RandomResize:
    def __init__(self, short_edge_length, max_size=sys.maxsize, sample_style='choice', interpolation=ttf.InterpolationMode.BILINEAR, log_name=''):
        assert sample_style in ['choice', 'range']
        assert isinstance(short_edge_length, (int, Sequence)), 'short_edge_length should be int or sequence. Got {}'.format(type(short_edge_length))

        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)

        self.short_edge_length = short_edge_length
        self.is_range = sample_style == 'range'
        self.max_size = max_size
        self.interpolation = interpolation

        logging.getLogger(log_name).info('{}/{}/{}'.format(self.__class__, self.short_edge_length, self.max_size))
        return

    def __call__(self, image: torch.Tensor):

        c, h, w = image.shape
        if self.is_range:
            size = random.randint(self.short_edge_length[0], self.short_edge_length[-1] + 1)
        else:
            size = random.choice(self.short_edge_length)

        scala = size * 1.0 / min(h, w)
        if h < w:
            new_h, new_w = size, w * scala
        else:
            new_h, new_w = h * scala, size

        if max(new_h, new_w) > self.max_size:
            scala = self.max_size * 1.0 / max(new_h, new_w)
            new_h, new_w = new_h * scala, new_w * scala

        new_h, new_w = int(new_h + 0.5), int(new_w + 0.5)
        return ttf.resize(image, size=[new_h, new_w], interpolation=self.interpolation), (new_h, new_w)

    def __str__(self):
        return 'RandomResize'


class RandomCrop:
    def __init__(self, f_min=0.7, f_max=0.9, step=0.1, log_name=''):
        assert 0 < f_min <= f_max <= 1.0
        self.crop_ratio = range_float(start=f_min, end=f_max, step=step, exclude=f_max+1.0)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.crop_ratio))
        return

    @staticmethod
    def get_crop_params(img: torch.Tensor, tw, th):
        c, h, w = img.shape

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()

        return i, j, th, tw

    def __call__(self, image: torch.Tensor, ratio_h=None, ratio_w=None):
        """
        :param image:
        :param ratio_h:
        :param ratio_w:
        :return:

        Returns:
            Cropped image and crop position (top, left, height, width)

        """
        ratio_h = ratio_h if ratio_h is not None else random.choice(self.crop_ratio)
        ratio_w = ratio_w if ratio_w is not None else random.choice(self.crop_ratio)

        c, h, w = image.shape
        th = int(h * ratio_h + 0.5)
        tw = int(w * ratio_w + 0.5)

        i, j, th, tw = RandomCrop.get_crop_params(image, tw=tw, th=th)

        return ttf.crop(image, i, j, th, tw), (i, j, th, tw)

    def __str__(self):
        return 'RandomCrop'


class RandomFlip:
    def __init__(self, log_name=''):
        self.method = [ttf.hflip, ttf.vflip]
        logging.getLogger(log_name).info('{}'.format(self.__class__))
        return

    def __call__(self, image: torch.Tensor):
        method = random.choice(self.method)
        return method(image), method

    def __str__(self):
        return 'RandomFlip'


class RandomRotate:
    def __init__(self, f_min=0, f_max=180, step=1, log_name='', interpolation=ttf.InterpolationMode.NEAREST):
        self.magnitude = [int(i) for i in range_float(f_min, f_max, step, 360)]
        self.interpolation = interpolation
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.magnitude))
        return

    def __call__(self, image: torch.Tensor):
        magnitude = random.choice(self.magnitude)
        return ttf.rotate(image, magnitude, interpolation=self.interpolation), magnitude

    def __str__(self):
        return 'RandomRotate'
