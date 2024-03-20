from abc import ABC
from typing import Any, Dict, Tuple, List, Union
import random
import numpy as np
import cv2
from . import functional as F
import math
from engine.transforms.build import BUILD_TRANSFORMER_REGISTRY

__all__ = [
    'RandomAffine',
    'RandomFlip',
    'Resize',
    'RandomResize',
    'RandomCrop',
    'RandomToneCurve',
    'RandomBrightnessContrast',
    'RandomGaussianBlur',
    'ToGray',
    'Normalize',
    'RandomColorJitter',
    'RandomBrightness',
    'RandomContrast',
    'RandomSaturation',
    'RandomHue',
    'RandomGamma',
    'RandomCLAHE',
    'RandomCompress',
    'RandomSharpen'
]


@BUILD_TRANSFORMER_REGISTRY.register()
class Pad32:
    """
    把图片pad到能被32整除
    """

    def __init__(self):
        return

    def __call__(self, results):
        height, width = results['img_shape']
        pad_h = math.ceil(height / 32) * 32
        pad_w = math.ceil(width / 32) * 32

        results['pad_shape'] = (pad_h, pad_w)

        self._pad_img(results)
        self._pad_box(results)
        self._pad_pts(results)

        return results

    def _pad_img(self, results):
        pad_h, pad_w = results['pad_shape']
        for key in results.get('img_fields', []):
            img = results[key]

            assert results['img_shape'] == img.shape[:-1]
            height, width = img.shape[0], img.shape[1]
            if (pad_h, pad_w) == (height, width):
                continue

            h_offset = pad_h - height
            w_offset = pad_w - width
            h_pad_top = h_offset // 2
            h_pad_bottom = h_offset - h_pad_top
            w_pad_left = w_offset // 2
            w_pad_right = w_offset - w_pad_left
            img = cv2.copyMakeBorder(img, top=h_pad_top, bottom=h_pad_bottom,
                                     left=w_pad_left, right=w_pad_right,
                                     borderType=cv2.BORDER_CONSTANT, value=0)
            results['pad_offset'] = (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right)

            results[key] = img

        results['img_shape'] = (pad_h, pad_w)
        return

    def _pad_box(self, results):
        """
        :param results:  results['box_fileds'] [x1, y1, x2, y2]
        :return:
        """
        height, width = results['img_shape']
        for key in results.get('box_fileds', []):
            bboxes = results[key]
            pad_top, pad_bottom, pad_left, pad_right = results.get('pad_offset', (0, 0, 0, 0))  # top, bottom, left, right
            bboxes = bboxes + [pad_left, pad_top, pad_left, pad_top]
            if self.clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height)
            results[key] = bboxes
        return

    def _pad_pts(self, results):
        """
        :param results:  results['pts_fields'] [x, y]
        :return: 
        """""
        height, width = results['img_shape']
        for key in results.get('pts_fields', []):
            pts = results[key]
            pad_top, pad_bottom, pad_left, pad_right = results.get('pad_offset', (0, 0, 0, 0))  # top, bottom, left, right
            pts = pts + [pad_left, pad_top, pad_left, pad_top]
            if self.clip_border:
                pts[:, 0] = np.clip(pts[:, 0], 0, width)
                pts[:, 1] = np.clip(pts[:, 1], 0, height)
            results[key] = pts
        return

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class Resize:
    """
    resize 到指定大小 target_size
    Args:
        interpolation funcional/INTER_CV_TYPE

    img_shape pad_offset 相关的都是 h w 形式
    """

    def __init__(self, target_size=0, interpolation='INTER_LINEAR', keep_ratio=True, is_padding=True, clip_border=True):
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.is_padding = is_padding
        self.clip_border = clip_border
        self.interpolation = list()
        inter_type = list(F.INTER_CV_TYPE.keys())
        if isinstance(interpolation, List) or isinstance(interpolation, Tuple):
            for inter in interpolation:
                if inter not in inter_type:
                    continue
                if inter in self.interpolation:
                    continue
                self.interpolation.append(inter)
            if len(self.interpolation) == 0:
                self.interpolation.append('INTER_LINEAR')
        else:
            self.interpolation.append(interpolation if interpolation in inter_type else 'INTER_LINEAR')

        if not self.keep_ratio:
            self.is_padding = False
        return

    def _get_new_size(self, target_size, results):
        height, width = results['img_shape']
        if self.keep_ratio:
            scale = float(target_size) / max(height, width)
            new_h, new_w = int(height * scale + 0.5), int(width * scale + 0.5)
            scale_w = scale
            scale_h = scale
        else:
            scale_w = target_size / width
            scale_h = target_size / height
            new_h, new_w = target_size, target_size

        results['img_shape'] = (new_h, new_w)
        results['scale'] = (scale_w, scale_h)
        results['keep_ratio'] = self.keep_ratio
        results['interpolation'] = random.choice(self.interpolation)
        return results

    def __call__(self, results):
        results = self._get_new_size(self.target_size, results)
        self._resize_img(results)
        self._resize_box(results)
        self._resize_pts(results)

        return results

    def _resize_img(self, results):
        new_img_shape = results['img_shape']
        interpolation = F.INTER_CV_TYPE[results['interpolation']]
        for key in results.get('img_fields', []):
            img = cv2.resize(results[key], dsize=(results['img_shape'][1], results['img_shape'][0]), interpolation=interpolation)
            if self.is_padding:
                height, width = img.shape[0], img.shape[1]
                if min(height, width) < self.target_size:
                    h_offset = self.target_size - height
                    w_offset = self.target_size - width
                    h_pad_top = h_offset // 2
                    h_pad_bottom = h_offset - h_pad_top
                    w_pad_left = w_offset // 2
                    w_pad_right = w_offset - w_pad_left
                    img = cv2.copyMakeBorder(img, top=h_pad_top, bottom=h_pad_bottom,
                                             left=w_pad_left, right=w_pad_right,
                                             borderType=cv2.BORDER_CONSTANT, value=0)
                    results['pad_offset'] = (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right)

            results[key] = img
            new_img_shape = img.shape[:2]

        results['img_shape'] = new_img_shape
        return

    def _resize_box(self, results):
        """
        :param results:  results['box_fileds'] [x1, y1, x2, y2]
        :return:
        """
        scale = results['scale']
        scales = [scale[0], scale[1], scale[0], scale[1]]
        height, width = results['img_shape']
        for key in results.get('box_fileds', []):
            bboxes = results[key] * scales
            pad_top, pad_bottom, pad_left, pad_right = results.get('pad_offset', (0, 0, 0, 0))  # top, bottom, left, right
            bboxes = bboxes + [pad_left, pad_top, pad_left, pad_top]
            if self.clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height)
            results[key] = bboxes
        return

    def _resize_pts(self, results):
        """
        :param results:  results['pts_fields'] [x, y]
        :return: 
        """""
        scale = results['scale']
        height, width = results['img_shape']
        for key in results.get('pts_fields', []):
            pts = results[key] * scale
            pad_top, pad_bottom, pad_left, pad_right = results.get('pad_offset', (0, 0, 0, 0))  # top, bottom, left, right
            pts = pts + [pad_left, pad_top]
            if self.clip_border:
                pts[:, 0] = np.clip(pts[:, 0], 0, width)
                pts[:, 1] = np.clip(pts[:, 1], 0, height)
            results[key] = pts
        return

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'target_size={self.target_size}'
        format_string += f', keep_ratio={self.keep_ratio}'
        format_string += f', is_padding={self.is_padding}'
        format_string += f', clip_border={self.clip_border}'
        format_string += f', interpolation={self.interpolation})'
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomAffine:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Args:
        rotate_degree_range (union[float,tuple(float), list(float)]): degrees of rotation transform.
            Default: 10.
        rotate_range (bool):  random.uniform(self.rotate_degree_range[0], self.rotate_degree_range[1]) if true else np.choice(rotate_degree_range)
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (1, 1).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 0.
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
        clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        border_ratio rotate
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 rotate_degree_range: Union[float, Tuple] = (-10, 10),
                 rotate_range=True,
                 max_translate_ratio=0,
                 scaling_ratio_range=(1., 1.),
                 max_shear_degree=0.0,
                 p=1.,
                 border_val=114,
                 clip_border=True,
                 min_bbox_size=2,
                 min_area_ratio=2.,
                 max_aspect_ratio=20,
                 border_ratio=1.2,
                 skip_filter=True,
                 ):
        assert 0 <= max_translate_ratio <= 1

        self.p = p

        self.rotate_range = rotate_range
        if self.rotate_range:  # random.uniform(self.rotate_degree_range[0], self.rotate_degree_range[1])
            assert (isinstance(rotate_degree_range, Tuple) or isinstance(rotate_degree_range, List)) and 2 == len(rotate_degree_range)
            self.rotate_degree_range = rotate_degree_range
        else:  # random.choice
            if isinstance(rotate_degree_range, Tuple) or isinstance(rotate_degree_range, List):
                self.rotate_degree_range = rotate_degree_range
            else:
                self.rotate_degree_range = (rotate_degree_range, rotate_degree_range)

        self.max_translate_ratio = max_translate_ratio
        self.border_val = border_val
        self.clip_border = clip_border
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.skip_filter = skip_filter
        self.border_ratio = border_ratio if border_ratio > 1 else 1
        return

    def __call__(self, results):
        if random.random() < self.p:
            img = results['img']
            height = img.shape[0]
            width = img.shape[1]
            # Rotation
            if not self.rotate_range:
                rotation_degree = random.choice(self.rotate_degree_range)
            else:
                rotation_degree = random.uniform(self.rotate_degree_range[0], self.rotate_degree_range[1])

            rotation_matrix = self._get_rotation_matrix(rotation_degree, width / 2, height / 2)

            # Translation
            trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
            trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height

            translate_matrix = self._get_translation_matrix(trans_x, trans_y)

            # Scaling
            scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                           self.scaling_ratio_range[1])
            scaling_matrix = self._get_scaling_matrix(scaling_ratio)

            # Shear
            x_degree = random.uniform(-self.max_shear_degree,
                                      self.max_shear_degree)
            y_degree = random.uniform(-self.max_shear_degree,
                                      self.max_shear_degree)
            shear_matrix = self._get_shear_matrix(x_degree, y_degree)

            warp_matrix = (
                    translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

            self._affine_img(results, warp_matrix)
            height, width = results['img_shape']
            for key in results.get('bbox_fields', []):
                results[key] = self._affine_box(results[key], warp_matrix, scaling_ratio, width, height)
            for key in results.get('pts_fields', []):
                results[key] = self._affine_point(results[key], warp_matrix, width, height)
        return results

    def _affine_img(self, results, affine_matrix):
        height, width = results['img_shape']
        width = int(width * self.border_ratio)
        height = int(height * self.border_ratio)
        for key in results.get('img_fields', []):
            results[key] = cv2.warpPerspective(
                results[key],
                affine_matrix,
                dsize=(width, height),
                borderValue=self.border_val)
        results['img_shape'] = (height, width)

    def _affine_box(self, bboxes, affine_matrix, scaling_ratio, width, height):
        """
        bboxes = [ N * 4]
        """
        num_bboxes = len(bboxes)
        if num_bboxes <= 0:
            return bboxes

        # bboxes = [0, 1, 2, 3] -> [x1, y1, x2, y2]
        # construct coordinate (x1, y1) (x2, y1) (x1, y2) (x2, y2)
        xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
        ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
        ones = np.ones_like(xs)
        # [[x1, x2, x3, x4, ...]
        # [y1, y2, y3, y4, ...]
        # [1,   1,  1,  1, ...]]
        points = np.vstack([xs, ys, ones])

        warp_points = affine_matrix @ points
        # normailze using the third value of homogeneous coordinates
        # (x, y, a) -> (x/a, y/a, 1)
        warp_points = warp_points[:2] / warp_points[2]
        xs = warp_points[0].reshape(num_bboxes, 4)
        ys = warp_points[1].reshape(num_bboxes, 4)

        warp_bboxes = np.vstack(
            (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

        if self.clip_border:
            warp_bboxes[:, [0, 2]] = \
                warp_bboxes[:, [0, 2]].clip(0, width)
            warp_bboxes[:, [1, 3]] = \
                warp_bboxes[:, [1, 3]].clip(0, height)

        # # remove outside bbox
        # valid_index = F.find_inside_bboxes(warp_bboxes, height, width)
        #
        # if not self.skip_filter:
        #     # filter bboxes
        #     filter_index = self.filter_gt_bboxes(
        #         bboxes * scaling_ratio, warp_bboxes)
        #     valid_index = valid_index & filter_index
        #
        # return warp_bboxes[valid_index]
        return warp_bboxes

    def _affine_point(self, pts, affine_matrix, width, height):
        """
        pts = [N * 2]
        """
        num_pts = len(pts)
        if num_pts <= 0:
            return pts

        # bboxes = [0, 1] -> [x1, y1]
        # construct coordinate (x1, y1)
        xs = pts[:, 0]
        ys = pts[:, 1]
        ones = np.ones_like(xs)
        points = np.vstack([xs, ys, ones])
        warp_points = affine_matrix @ points
        warp_points = warp_points[:2] / warp_points[2]
        warp_pts = warp_points.T

        if self.clip_border:
            warp_pts[:, 0] = warp_pts[:, 0].clip(0, width)
            warp_pts[:, 1] = warp_pts[:, 1].clip(0, height)

        # # remove outside pts
        # valid_index = F.find_inside_pts(warp_pts, height, width)
        # return warp_pts[valid_index]
        return warp_pts

    def filter_gt_bboxes(self, origin_bboxes, wrapped_bboxes):
        origin_w = origin_bboxes[:, 2] - origin_bboxes[:, 0]
        origin_h = origin_bboxes[:, 3] - origin_bboxes[:, 1]
        wrapped_w = wrapped_bboxes[:, 2] - wrapped_bboxes[:, 0]
        wrapped_h = wrapped_bboxes[:, 3] - wrapped_bboxes[:, 1]
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    @staticmethod
    def _get_rotation_matrix(degree, cx=0, cy=0):
        radian = math.radians(degree)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), -cx * np.cos(radian) + cy * np.sin(radian) + cx],
             [np.sin(radian), np.cos(radian), -cx * np.sin(radian) - cy * np.cos(radian) + cy],
             [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_share_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomFlip:
    def __init__(self, p, direction='horizontal'):
        assert 0 <= p <= 1
        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
            direction = [direction]
        elif isinstance(direction, list):
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.p = p
        self.direction = direction
        return

    def __call__(self, results):
        if random.random() < self.p:
            direction = random.choice(self.direction)
            results['direction'] = direction

            for key in results['img_fields']:
                results[key] = F.imflip(results[key], direction)

            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key], results['img_shape'], direction)

            for key in results.get('pts_fields', []):
                results[key] = self.pts_flip(results[key], results['img_shape'], direction)

        return results

    def bbox_flip(self, bboxes, img_shape, direction):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def pts_flip(self, pts, img_shape, direction):
        assert pts.shape[-1] % 2 == 0
        flipped = pts.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0] = w - pts[..., 0]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1] = h - pts[..., 1]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0] = w - pts[..., 0]
            flipped[..., 1] = h - pts[..., 1]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '[hflip vflip] p={0})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomResize(Resize):
    """
    先随机resize到指定大小，再padding到target_size
    """

    def __init__(self, max_edge_length, padding_size, interpolation='INTER_LINEAR', keep_ratio=True, clip_border=True):
        super(RandomResize, self).__init__(padding_size, interpolation, keep_ratio, True, clip_border)

        if isinstance(max_edge_length, int):
            max_edge_length = (max_edge_length, max_edge_length)
        self.max_edge_length = [length for length in max_edge_length if length < padding_size]
        self.max_edge_length.append(padding_size)
        self.padding_size = padding_size
        return

    def __call__(self, results):
        new_size = random.choice(self.max_edge_length)
        results = self._get_new_size(new_size, results)
        self._resize_img(results)
        self._resize_box(results)
        self._resize_pts(results)

        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'max_edge_length={self.max_edge_length}'
        format_string += f', padding_size={self.padding_size}'
        format_string += f', keep_ratio={self.keep_ratio}'
        format_string += f', clip_border={self.clip_border}'
        format_string += f', interpolation={self.interpolation})'
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomCrop:
    """
    和 RandomResizeCrop 不同的是，只会产生很小的黑框
    """

    def __init__(self, min_crop_ratio, max_crop_ratio, crop_step=0.1, p=1., clip_border=True):
        assert 0 < min_crop_ratio <= max_crop_ratio <= 1
        self.crop_ratio = F.range_float(1 - max_crop_ratio, 1 - min_crop_ratio, crop_step, exclude=max_crop_ratio + 1)
        self.p = p
        self.clip_border = clip_border
        return

    def _get_crop_params(self, width, height):

        ratio_h = random.choice(self.crop_ratio)
        ratio_w = random.choice(self.crop_ratio)

        target_height = int(height * ratio_h + 0.5)
        target_width = int(width * ratio_w + 0.5)

        if height + 1 < target_height or width + 1 < target_width:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((target_height, target_width), (height, width))
            )

        if target_width == width and target_height == height:
            return 0, 0, width, height

        x1 = random.randint(0, width - target_width + 1)
        y1 = random.randint(0, height - target_height + 1)

        return x1, y1, target_width, target_height

    def __call__(self, results):
        if random.random() < self.p:
            img = results['img']
            height = img.shape[0]
            width = img.shape[1]
            x1, y1, target_width, target_height = self._get_crop_params(height=height, width=width)

            for key in results['img_fields']:
                results[key] = results[key][y1:target_height + y1, x1:target_width + x1, ...]
            results['img_shape'] = (target_height, target_width)

            self._crop_bboxes(results, x1, y1)
            self._crop_pts(results, x1, y1)

        return results

    def _crop_bboxes(self, results, offset_x, offset_y):
        height, width = results['img_shape']

        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height)

            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            results[key] = bboxes[valid_inds, :]
        return

    def _crop_pts(self, results, offset_x, offset_y):
        height, width = results['img_shape']

        for key in results.get('pts_fields', []):
            pts_offset = np.array([offset_x, offset_y], dtype=np.float32)
            pts = results[key] - pts_offset
            if self.clip_border:
                pts[:, 0] = np.clip(pts[:, 0], 0, width)
                pts[:, 1] = np.clip(pts[:, 1], 0, height)
            results[key] = pts
        return

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'crop_ratio={self.crop_ratio}'
        format_string += f', clip_border={self.clip_border}'
        format_string += f', p={self.p})'
        return format_string


class BasicColorTransform(ABC):

    def __init__(self, p: float = 1.0):
        self.p = p
        return

    def __call__(self, results):
        if random.random() < self.p:
            params = self.get_params(results)
            for key in results['color_fields']:
                results[key] = self.apply(results[key], **params)
        return results

    def get_params(self, results) -> Dict[str, Any]:
        return dict()

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomGaussianBlur(BasicColorTransform):
    def __init__(self, blur_limit=(3, 7), sigma_limit=0, p: float = 0.5):
        super(RandomGaussianBlur, self).__init__(p)
        self.blur_limit = F.to_tuple(blur_limit, 0)
        self.sigma_limit = F.to_tuple(sigma_limit if sigma_limit is not None else 0, 0)

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            print(
                "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                "blur_limit minimum value changed to 3."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
                self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("GaussianBlur supports only odd blur limits.")

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.gaussian_blur(img, params['ksize'], sigma=params['sigma'])

    def get_params(self, kwargs) -> Dict[str, float]:
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize != 0 and ksize % 2 != 1:
            ksize = (ksize + 1) % (self.blur_limit[1] + 1)

        return {"ksize": ksize, "sigma": random.uniform(*self.sigma_limit)}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'blur_limit={}, '.format(self.blur_limit)
        format_string += 'sigma_limit={}, '.format(self.sigma_limit)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomBrightnessContrast(BasicColorTransform):
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5):
        """
        :param brightness_limit: disable brightness when  brightness_limit = 0
        :param contrast_limit: disable contrast when  contrast_limit = 0
        :param brightness_by_max:
        :param p:
        """
        super(RandomBrightnessContrast, self).__init__(p)
        self.brightness_limit = F.to_tuple(brightness_limit)
        self.contrast_limit = F.to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.brightness_contrast_adjust(img, params['alpha'], params['beta'], self.brightness_by_max)

    def get_params(self, kwargs):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness_limit={}, '.format(self.brightness_limit)
        format_string += 'contrast_limit={}, '.format(self.contrast_limit)
        format_string += 'brightness_by_max={}, '.format(self.brightness_by_max)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomToneCurve(BasicColorTransform):
    def __init__(self, scale=0.1, p=0.5):
        """
        :param scale: standard deviation of the normal distribution.
                      Used to sample random distances to move two control points that modify the image's curve.
                      Values should be in range [0, 1]. Default: 0.1
        :param p:
        """
        super(RandomToneCurve, self).__init__(p, )
        self.scale = scale
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.move_tone_curve(img, params['low_y'], params['high_y'])

    def get_params(self, kwargs):
        return {
            "low_y": np.clip(np.random.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y": np.clip(np.random.normal(loc=0.75, scale=self.scale), 0, 1),
        }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'scale={}, '.format(self.scale)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomBrightness(BasicColorTransform):
    def __init__(self, brightness_limit=0.2, p=0.5):
        """
        :param brightness_limit: disable brightness when  brightness_limit = 0
        :param p:
        """
        super(RandomBrightness, self).__init__(p, )
        self.brightness_limit = F.check_values(brightness_limit, 'brightness')
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.adjust_brightness(img, params['factor'])

    def get_params(self, kwargs):
        return {
            "factor": random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness_limit={}, '.format(self.brightness_limit)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomContrast(BasicColorTransform):
    def __init__(self, contrast_limit=0.2, p=0.5):
        """
        :param contrast_limit: disable brightness when  contrast_limit = 0
        :param p:
        """
        super(RandomContrast, self).__init__(p, )
        self.contrast_limit = F.check_values(contrast_limit, 'contrast')
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.adjust_contrast(img, params['factor'])

    def get_params(self, kwargs):
        return {
            "factor": random.uniform(self.contrast_limit[0], self.contrast_limit[1])
        }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'contrast_limit={}, '.format(self.contrast_limit)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomSaturation(BasicColorTransform):
    def __init__(self, saturation_limit=0.2, p=0.5):
        """
        :param saturation_limit: disable brightness when  saturation_limit = 0
        :param p:
        """
        super(RandomSaturation, self).__init__(p, )
        self.saturation_limit = F.check_values(saturation_limit, 'saturation')
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.adjust_saturation(img, params['factor'], params.get('gamma', 0))

    def get_params(self, kwargs):
        return {
            "factor": random.uniform(self.saturation_limit[0], self.saturation_limit[1]),
        }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'saturation_limit={}, '.format(self.saturation_limit)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomHue(BasicColorTransform):
    def __init__(self, hue_limit=0.2, p=0.5):
        """
        :param hue_limit: disable brightness when  hue_limit = 0
        :param p:
        """
        super(RandomHue, self).__init__(p, )
        self.hue_limit = F.check_values(hue_limit, 'hue', offset=0, bounds=[-0.5, +0.5], clip=False)
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.adjust_hue(img, params['factor'])

    def get_params(self, kwargs):
        return {
            "factor": random.uniform(self.hue_limit[0], self.hue_limit[1])
        }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'hue_limit={}, '.format(self.hue_limit)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomGamma(BasicColorTransform):
    def __init__(self, gamma_limit=(0.8, 1.2), p=0.5):
        """
        :param gamma_limit:  (float or (float, float)): If gamma_limit is a single float value,
            the range will be (-gamma_limit, gamma_limit). Default: (0.8, 1.2).
        :param p:
        """

        super(RandomGamma, self).__init__(p, )
        self.gamma_limit = F.to_tuple(gamma_limit)
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.gamma_transform(img, gamma=params['gamma'])

    def get_params(self, kwargs):
        return {'gamma': random.uniform(self.gamma_limit[0], self.gamma_limit[1])}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'gamma_limit={}, '.format(self.gamma_limit)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomSharpen(BasicColorTransform):
    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5):
        """

        :param alpha: range to choose the visibility of the sharpened image. At 0, only the original image is
                      visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        :param lightness:  range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        :param p:
        """
        super(RandomSharpen, self).__init__(p, )
        self.alpha = self.__check_values(F.to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.lightness = self.__check_values(F.to_tuple(lightness, 0.0), name="lightness")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
        return value

    @staticmethod
    def __generate_sharpening_matrix(alpha_sample, lightness_sample):
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness_sample, -1], [-1, -1, -1]],
            dtype=np.float32,
        )

        matrix = (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return matrix

    def get_params(self, kwargs):
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)
        sharpening_matrix = self.__generate_sharpening_matrix(alpha_sample=alpha, lightness_sample=lightness)
        return {'sharpening_matrix': sharpening_matrix}

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.convolve(img, params['sharpening_matrix'])

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'alpha={}, '.format(self.alpha)
        format_string += 'lightness={}, '.format(self.lightness)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class ToGray(BasicColorTransform):
    def __init__(self, p):
        super(ToGray, self).__init__(p, )
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if F.is_grayscale_image(img):
            print("The image is already gray.")
            return img
        if not F.is_rgb_image(img):
            raise TypeError("ToGray transformation expects 3-channel images.")

        return F.to_gray(img)

    def __repr__(self):
        return self.__class__.__name__


@BUILD_TRANSFORMER_REGISTRY.register()
class Normalize(BasicColorTransform):
    """
    Args:
        mean (list of float): mean values for each channel.
        std  (list of float): std values for each channel.
        max_pixel_value:
    Targets:
        image [np.float]

    Image types:
        uint8, float32
    """

    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 max_pixel_value=255):
        super(Normalize, self).__init__(p=1)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.normalize(img, self.mean, self.std, self.max_pixel_value)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'std={}, '.format(self.std)
        format_string += 'mean={}, '.format(self.mean)
        format_string += 'max_pixel_value={})'.format(self.max_pixel_value)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomCLAHE(BasicColorTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        super(RandomCLAHE, self).__init__(p)
        self.clip_limit = F.to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)
        return

    def apply(self, img, **params):
        if not F.is_rgb_image(img) and not F.is_grayscale_image(img):
            print("CLAHE transformation expects 1-channel or 3-channel images.")
            return img

        return F.clahe(img, params['clip_limit'], self.tile_grid_size)

    def get_params(self, kwargs):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'clip_limit={}, '.format(self.clip_limit)
        format_string += 'tile_grid_size={}, '.format(self.tile_grid_size)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomCompress(BasicColorTransform):
    """Decreases image quality by Jpeg, WebP compression of an image.

    Args:
        quality_lower (float): lower bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper (float): upper bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG

    Targets:
        image

    Image types:
        uint8, float32
    """
    JPEG = 'jpg'
    WEBP = 'webp'

    def __init__(self, quality_lower=99, quality_upper=100, quality_step=1, compression_type='jpg', p=0.5):
        super(RandomCompress, self).__init__(p=p)
        assert compression_type in [self.JPEG, self.WEBP], f'{compression_type} not in [jpg, webp]'

        low_thresh_quality_assert = 1 if compression_type == self.WEBP else 0
        if not low_thresh_quality_assert <= quality_lower <= 100:
            raise ValueError("Invalid quality_lower. Got: {}".format(quality_lower))
        if not low_thresh_quality_assert <= quality_upper <= 100:
            raise ValueError("Invalid quality_upper. Got: {}".format(quality_upper))

        self.quality = [int(v) for v in F.range_float(quality_lower, quality_upper, quality_step, quality_upper + quality_lower)]

        self.img_type = f'.{self.WEBP}' if compression_type == self.WEBP else f'.{self.JPEG}'
        return

    def apply(self, img, **params):
        if not F.is_rgb_image(img) and not F.is_rgba_image(img) and not F.is_grayscale_image(img):
            print("Compress transformation expects 1-channel or 3-channel, 4-channel images.")
            return img

        return F.image_compression(img, params['quality'], self.img_type)

    def get_params(self, kwargs):
        return {'quality': random.choice(self.quality)}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'quality={}, '.format(self.quality)
        format_string += 'compression_type={}, '.format(self.img_type)
        format_string += 'p={})'.format(self.p)
        return format_string


@BUILD_TRANSFORMER_REGISTRY.register()
class RandomColorJitter:
    def __init__(self,
                 brightness_limit=0.2, brightness_p=0.5,
                 contrast_limit=0.2, contrast_p=0.5,
                 saturation_limit=0.2, saturation_p=0.5,
                 hue_limit=0.1, hue_p=0.1,
                 blur_limit=(3, 7), sigma_limit=0, blur_p=0.2,
                 gamma_limit=(0.3, 2.0), gamma_p=0.5,
                 clahe_limit=4.0, clahe_p=0.2,
                 ):

        self.jitter = [
            RandomBrightness(brightness_limit, brightness_p),
            RandomContrast(contrast_limit, contrast_p),
            RandomSaturation(saturation_limit, saturation_p),
            RandomHue(hue_limit, hue_p),
            RandomGaussianBlur(blur_limit, sigma_limit, blur_p),
            RandomGamma(gamma_limit, gamma_p),
            RandomCLAHE(clip_limit=clahe_limit, p=clahe_p),
        ]

        return

    def __call__(self, results):
        jitter_index = [i for i in range(len(self.jitter))]
        random.shuffle(jitter_index)
        for idx in jitter_index:
            results = self.jitter[idx](results)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for jitter in self.jitter[:-1]:
            format_string += '{}, '.format(jitter)
        format_string += '{})'.format(self.jitter[-1])
        return format_string
