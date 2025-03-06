import random
from typing import List, Dict
import numpy as np
import cv2

from engine.transforms import Resize

"""
 针对目标检测
 数据格式必须是voc,
 且这类增强必须在第一个
"""

__all__ = [
    'MosaicTransform'
]


class BaseMixTransform:

    def __init__(self, dataset: List, p=0.0, max_size=640) -> None:
        """
        Args:
            dataset: 数据格式 (img_path:str, bboxes:np.ndarray, bboxes_cls:list)
            p: 执行的概率
            max_size: 执行mix增强的所有图片指定缩放到max_size，保证每个图片大小一致
        """
        self.dataset = dataset
        self.p = p
        self.imgsz = max_size
        self.resize_fn = Resize(target_size=max_size, is_padding=False)
        return

    def __call__(self, label):
        """
        Args:
            label:
                dict(
                    color_fields=['img'],
                    img_fields=['img', 'mask'],
                    pts_fields=['pts'],
                    bbox_fields=['bbox'],
                    pad_value=dict(img=0, mask=255, pts=0, bbox=0),
                    img=np.ndarray,
                    mask=np.ndarray,
                    pts=np.ndarray,
                    bbox=np.ndarray
                    img_shape = ori_img.shape[:2],
                    ori_shape = ori_img.shape[:2]，
                    pad_offset = [top, bottom, left, right],
                    scale = [w, h],
                    bbox_cls_fields=[1]
                )

        Returns:
            返回格式同 label

        """
        if random.uniform(0, 1) > self.p:
            return label

        if label.get('bbox_fields', None) is not None:
            assert len(label['bbox_fields']) == 1, 'the bbox_fields size must be equal 1'
        if label.get('pts_fields', None) is not None:
            assert len(label['pts_fields']) == 1, 'the pts_fields size must be equal 1'

        # Get images information will be used for Mosaic or MixUp
        mix_labels = self.get_mix_labels(label['bbox_fields'][0], label['bbox_cls_fields'][0])

        label = self.resize_fn(label)
        for i, data in enumerate(mix_labels):
            mix_labels[i] = self.resize_fn(data)

        label['mix_labels'] = mix_labels

        # Mosaic or MixUp
        label = self._mix_transform(label)
        label.pop('mix_labels', None)
        return label

    def _mix_transform(self, labels):
        """
        Args:
            labels:

        Returns:

        """
        raise NotImplementedError

    def get_mix_labels(self, bbox_field, bbox_cls_fields) -> List[Dict]:
        """
        Gets a list of shuffled indexes for mosaic augmentation.

        Returns:
            (List[Dict]): A list of shuffled indexes from the dataset.
            result = [
                dict(
                    color_fields=['img'],
                    img_fields=['img', 'mask'],
                    pts_fields=['pts'],
                    bbox_fields=['bbox'],
                    pad_value=dict(img=0, mask=255, pts=0, bbox=0),
                    img=np.ndarray,
                    mask=np.ndarray,
                    pts=np.ndarray,
                    bbox=np.ndarray
                    img_shape = ori_img.shape[:2],
                    ori_shape = ori_img.shape[:2]，
                    pad_offset = [top, bottom, left, right],
                    scale = [w, h],
                    bbox_cls_fields=[1]
                )
            ]
        """
        raise NotImplementedError

    @staticmethod
    def _update_labels(labels, offset_x, offset_y, border):
        """
        Args:
            labels (Dict): A dictionary containing image and instance information.
            offset_x (int): Padding width to be added to the x-coordinates.
            offset_y (int): Padding height to be added to the y-coordinates.
            border (Tuple): h,w

        Returns:
            (Dict): Updated labels dictionary with adjusted instance coordinates.
        """
        for key in labels.get('bbox_fields', []):
            bbox_offset = np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
            labels[key] = labels[key] + bbox_offset
            labels[key][:, 0::2] = np.clip(labels[key][:, 0::2], 0, border[1])
            labels[key][:, 1::2] = np.clip(labels[key][:, 1::2], 0, border[0])

        for key in labels.get('pts_fields', []):
            pts_offset = np.array([offset_x, offset_y], dtype=np.float32)
            labels[key] = labels[key] + pts_offset
            labels[key][:, 0] = np.clip(labels[key][:, 0], 0, border[1])
            labels[key][:, 1] = np.clip(labels[key][:, 1], 0, border[0])
        return labels


class MosaicTransform(BaseMixTransform):
    """
    Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (Tuple[int, int]): Border size for width and height.

    Methods:
        get_indexes: Returns a list of random indexes from the dataset.
        _mix_transform: Applies mixup transformation to the input image and labels.
        _mosaic3: Creates a 1x3 image mosaic.
        _mosaic4: Creates a 2x2 image mosaic.
        _mosaic9: Creates a 3x3 image mosaic.
        _update_labels: Updates labels with padding.
        _cat_labels: Concatenates labels and clips mosaic border instances.
    """

    def __init__(self, dataset: List, max_size=640, p=1.0, n=4, border_step=2):
        """
        Initializes the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (List): The dataset (img_path, bboxes, bboxes_cls).
            max_size (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
            border_step (int) :
        """
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        assert n in {3, 4, 9}, 'grid must be equal to 4 or 9.'
        assert border_step > 0, f'the border_step {border_step} should be > 0'
        super().__init__(dataset=dataset, p=p, max_size=max_size)
        self.border = (-max_size // border_step, -max_size // border_step)  # width, height
        self.n = n
        return

    def get_mix_labels(self, bbox_field, bbox_cls_field) -> List[Dict]:
        """
        Returns a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.

        Returns:
            (List[Dict]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).
        """
        index = [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

        mix_result = list()
        for i in index:
            img_path, bboxes, bboxes_cls = self.dataset[i]
            ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            result = dict(
                img=ori_img,
                img_fields=['img'],
                color_fields=['img'],
                ori_shape=ori_img.shape[:2],
                img_shape=ori_img.shape[:2],
            )

            if bboxes.size > 0:
                result['bbox_fields'] = [bbox_field]
                result['bbox_cls_fields'] = [bbox_cls_field]
                result[bbox_field] = bboxes
                result[bbox_cls_field] = bboxes_cls

            mix_result.append(result)
        return mix_result

    def _mix_transform(self, labels):
        """
        Applies mosaic augmentation to the input image and labels.

        This method combines multiple images (3, 4, or 9) into a single mosaic image based on the 'n' attribute.
        It ensures that rectangular annotations are not present and that there are other images available for
        mosaic augmentation.

        Args:
            labels (Dict): A dictionary containing image data and annotations:
                - 'mix_labels': A list of dictionaries containing data for other images to be used in the mosaic.

        Returns:
            (Dict): A dictionary containing the mosaic-augmented image and updated annotations.
        """
        assert len(labels.get('mix_labels', [])), 'There are no other images for mosaic augment.'
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # This code is modified for mosaic3 method.

    def _mosaic3(self, labels):
        """
        Creates a 1x3 image mosaic by combining three images.

        This method arranges three images in a horizontal layout, with the main image in the center and two
        additional images on either side. It's part of the Mosaic augmentation technique used in object detection.

        Args:
            labels (Dict): A dictionary containing image and label information for the main (center) image.
                Must include 'img' key with the image array, and 'mix_labels' key with a list of two
                dictionaries containing information for the side images.

        Returns:
            (Dict): A dictionary with the mosaic image and updated labels. Keys include:
                - 'img' (np.ndarray): The mosaic image array with shape (H, W, C).
                - Other keys from the input labels, updated to reflect the new image dimensions.
        """
        mosaic_labels = []
        s = self.imgsz
        h0, w0 = 0, 0
        img3: np.ndarray = None
        for i in range(3):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('img_shape')
            # Place img in img3
            if i == 0:  # center
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 3 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # right
                c = s + w0, s, s + w0 + w, s + h
            else:  # left
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates

            img3[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch,
                                               padw + self.border[0],
                                               padh + self.border[1],
                                               img3[-self.border[0]: self.border[0], -self.border[1]: self.border[1]].shape[:2])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels['img'] = img3[-self.border[0]: self.border[0], -self.border[1]: self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        """
        Creates a 2x2 image mosaic from four input images.

        This method combines four images into a single mosaic image by placing them in a 2x2 grid. It also
        updates the corresponding labels for each image in the mosaic.

        Args:
            labels (Dict): A dictionary containing image data and labels for the base image (index 0) and three
                additional images (indices 1-3) in the 'mix_labels' key.

        Returns:
            (Dict): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
                image as a numpy array, and other keys contain the combined and adjusted labels for all four images.
        """
        mosaic_labels = []
        s = self.imgsz
        img4: np.ndarray = None
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch['img_shape']

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh, img4.shape[:2])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        final_labels['img_shape'] = img4.shape[:2]
        return final_labels

    def _mosaic9(self, labels):
        """
        Creates a 3x3 image mosaic from the input image and eight additional images.

        This method combines nine images into a single mosaic image. The input image is placed at the center,
        and eight additional images from the dataset are placed around it in a 3x3 grid pattern.

        Args:
            labels (Dict): A dictionary containing the input image and its associated labels. It should have
                the following keys:
                - 'img' (numpy.ndarray): The input image.
                - 'img_shape' (Tuple[int, int]): The shape of the resized image (height, width).
                - 'mix_labels' (List[Dict]): A list of dictionaries containing information for the additional
                  eight images, each with the same structure as the input labels.

        Returns:
            (Dict): A dictionary containing the mosaic image and updated labels. It includes the following keys:
                - 'img' (numpy.ndarray): The final mosaic image.
                - Other keys from the input labels, updated to reflect the new mosaic arrangement.
        """
        mosaic_labels = []
        s = self.imgsz
        img9: np.ndarray = None
        h0, w0 = 0, 0
        hp, wp = -1, -1  # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('img_shape')

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            else:  # i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch,
                                               padw + self.border[0],
                                               padh + self.border[1],
                                               img9[-self.border[0]: self.border[0], -self.border[1]: self.border[1]].shape[:2])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels['img'] = img9[-self.border[0]: self.border[0], -self.border[1]: self.border[1]]
        return final_labels

    def _cat_labels(self, mosaic_labels):
        """
        Concatenates and processes labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes.

        Args:
            mosaic_labels (List[Dict]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (Dict): A dictionary containing concatenated and processed labels for the mosaic image, including:
                    - color_fields=['img'],
                    - img_fields=['img', 'mask'],
                    - pts_fields=['pts'],
                    - bbox_fields=['bbox'],
                    - pad_value=dict(img=0, mask=255, pts=0, bbox=0),
                    - img=np.ndarray,
                    - mask=np.ndarray,
                    - pts=np.ndarray,
                    - bbox=np.ndarray
                    - img_shape = ori_img.shape[:2],
                    - ori_shape = ori_img.shape[:2]，
                    - pad_offset = [top, bottom, left, right],
                    - scale = [w, h],
                    - bbox_cls_fields=[1]
        """
        if len(mosaic_labels) == 0:
            return {}

        cls = []
        bbox = []
        pts = []
        imgsz = self.imgsz * 2  # mosaic imgsz

        # Final labels
        final_labels = {
            'ori_shape': mosaic_labels[0]['ori_shape'],
            'resized_shape': (imgsz, imgsz),
            'mosaic_border': self.border,
            'mosaic_nums': self.n,
            'color_fields': mosaic_labels[0]['color_fields']
        }

        for labels in mosaic_labels:
            if labels.get('bbox_fields', None) is not None:
                bbox.append(labels[labels['bbox_fields'][0]])
                cls.extend(labels[labels['bbox_cls_fields'][0]])
            if labels.get('pts_fields', None) is not None:
                pts.append(labels[labels['pts_fields'][0]])

        if len(bbox) > 0:
            final_labels[mosaic_labels[0]['bbox_cls_fields'][0]] = cls
            final_labels[mosaic_labels[0]['bbox_fields'][0]] = np.concatenate(bbox, axis=0)
        if len(pts) > 0:
            final_labels[mosaic_labels[0]['pts_fields'][0]] = np.concatenate(pts, axis=0)
        return final_labels
