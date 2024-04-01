from typing import Sequence, Union
import numbers
from itertools import product
import numpy as np
import cv2
import torch

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

INTER_CV_TYPE = dict(
    INTER_AREA=3,
    INTER_BITS=5,
    INTER_BITS2=10,
    INTER_CUBIC=2,
    INTER_LANCZOS4=4,
    INTER_LINEAR=1,
    INTER_LINEAR_EXACT=5,
    INTER_MAX=7,
    INTER_NEAREST=0,
    INTER_NEAREST_EXACT=6,
)


def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def clip(img: np.ndarray, dtype: np.dtype, maxval: float) -> np.ndarray:
    return np.clip(img, 0, maxval).astype(dtype)


def is_grayscale_image(image: np.ndarray) -> bool:
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


def is_rgb_image(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[-1] == 3


def is_rgba_image(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[-1] == 4


def _is_numpy_image(img: np.ndarray) -> bool:
    return img.ndim in {2, 3}


def range_float(start, end, step, exclude) -> list:
    assert start <= end
    if step < 1:
        step_str = str(step)
        arr = step_str.split('.')
        assert len(arr[1][arr[1].rfind('0') + 1:]) == 1, 'the step must one float, but the step={}'.format(step)
        zero_cnt = len(arr[1])
        base = pow(10, zero_cnt)
    else:
        base = 1
    return [i / base for i in range(*{'start': int(start * base), 'stop': int(end * base + 1), 'step': int(step * base)}.values()) if i != int(exclude * base)]


def to_tuple(param, low=None):
    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        if len(param) != 2:
            raise ValueError("to_tuple expects 1 or 2 values")
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    return tuple(param)


def check_values(value, name, offset=1, bounds=(0, float("inf")), clip=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError("If {} is a single number, it must be non negative.".format(name))
        value = [offset - value, offset + value]
        if clip:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
    else:
        raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

    return value


def pad(img: np.ndarray, t_h: int, t_w: int = -1) -> np.ndarray:
    if t_w < 0:
        t_w = t_h

    h, w = img.shape[0], img.shape[1]
    if h >= t_h and w >= t_w:
        return img

    h_offset = t_h - h
    w_offset = t_w - w
    h_pad_top = h_offset // 2
    h_pad_bottom = h_offset - h_pad_top
    w_pad_left = w_offset // 2
    w_pad_right = w_offset - w_pad_left

    return cv2.copyMakeBorder(img, top=h_pad_top, bottom=h_pad_bottom, left=w_pad_left, right=w_pad_right, borderType=cv2.BORDER_CONSTANT, value=0)


# --------------none geometric transformation---------------------- #
def gaussian_blur(img: np.ndarray, ksize: int, sigma: float = 0) -> np.ndarray:
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    return cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)


def defocus(img: np.ndarray, radius: int, alias_blur: float) -> np.ndarray:
    length = np.arange(-max(8, radius), max(8, radius) + 1)
    ksize = 3 if radius <= 8 else 5

    x, y = np.meshgrid(length, length)
    aliased_disk = np.array((x ** 2 + y ** 2) <= radius ** 2, dtype=np.float32)
    aliased_disk /= np.sum(aliased_disk)

    kernel = gaussian_blur(aliased_disk, ksize, sigma=alias_blur)
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)


def glass_blur(
        img: np.ndarray, sigma: float, max_delta: int, iterations: int, dxy: np.ndarray, mode: str
) -> np.ndarray:
    x = cv2.GaussianBlur(np.array(img), sigmaX=sigma, ksize=(0, 0))

    if mode == "fast":
        hs = np.arange(img.shape[0] - max_delta, max_delta, -1)
        ws = np.arange(img.shape[1] - max_delta, max_delta, -1)
        h: Union[int, np.ndarray] = np.tile(hs, ws.shape[0])
        w: Union[int, np.ndarray] = np.repeat(ws, hs.shape[0])

        for i in range(iterations):
            dy = dxy[:, i, 0]
            dx = dxy[:, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]

    elif mode == "exact":
        for ind, (i, h, w) in enumerate(
                product(
                    range(iterations),
                    range(img.shape[0] - max_delta, max_delta, -1),
                    range(img.shape[1] - max_delta, max_delta, -1),
                )
        ):
            ind = ind if ind < len(dxy) else ind % len(dxy)
            dy = dxy[ind, i, 0]
            dx = dxy[ind, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]
    else:
        ValueError(f"Unsupported mode `{mode}`. Supports only `fast` and `exact`.")

    return cv2.GaussianBlur(x, sigmaX=sigma, ksize=(0, 0))


def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if img.dtype == np.float32 and ksize not in {3, 5}:
        raise ValueError(f"Invalid ksize value {ksize}. For a float32 image the only valid ksize values are 3 and 5")

    return cv2.medianBlur(img, ksize=ksize)


def blur(img: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.blur(img, ksize=(ksize, ksize))


def convolve(img: np.ndarray, kernel):
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)


def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)

    return img


def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += (alpha * beta) * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


def _adjust_brightness_uint8(img, factor):
    lut = np.arange(0, 256) * factor
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def adjust_brightness(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    if img.dtype == np.uint8:
        return _adjust_brightness_uint8(img, factor)

    return clip(img * factor, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_contrast_uint8(img, factor, mean):
    lut = np.arange(0, 256) * factor
    lut = lut + mean * (1 - factor)
    lut = clip(lut, img.dtype, 255)

    return cv2.LUT(img, lut)


def adjust_contrast(img, factor):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    if img.dtype == np.uint8:
        return _adjust_contrast_uint8(img, factor, mean)

    return clip(
        img.astype(np.float32) * factor + mean * (1 - factor),
        img.dtype,
        MAX_VALUES_BY_DTYPE[img.dtype],
    )


def adjust_saturation(img, factor, gamma=0):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    if img.dtype == np.uint8:
        return result

    # OpenCV does not clip values for float dtype
    return clip(result, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_hue_uint8(img, factor):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = cv2.LUT(img[..., 0], lut)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue(img, factor):
    if is_grayscale_image(img):
        return img

    if factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


def move_tone_curve(img, low_y, high_y):
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        img (numpy.ndarray): RGB or grayscale image.
        low_y (float): y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float): y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
    """
    input_dtype = img.dtype

    if low_y < 0 or low_y > 1:
        raise ValueError("low_shift must be in range [0, 1]")
    if high_y < 0 or high_y > 1:
        raise ValueError("high_shift must be in range [0, 1]")

    if input_dtype != np.uint8:
        raise ValueError("Unsupported image type {}".format(input_dtype))

    t = np.linspace(0.0, 1.0, 256)

    # Defines responze of a four-point bezier curve
    def evaluate_bez(t):
        return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t ** 2 * high_y + t ** 3

    evaluate_bez = np.vectorize(evaluate_bez)
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)

    return cv2.LUT(img, lut=remapping)


def to_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def to_tensor(data: np.ndarray):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    """
    if len(data.shape) < 3:
        data = np.expand_dims(data, -1)
    data = np.ascontiguousarray(data.transpose(2, 0, 1))

    return torch.from_numpy(data)


def find_inside_bboxes(bboxes, img_h, img_w):
    """Find bboxes as long as a part of bboxes is inside the image.

    Args:
        bboxes (Tensor): Shape (N, 4).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        Tensor: Index of the remaining bboxes.
    """
    inside_inds = (bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0) \
                  & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0)
    return inside_inds


def find_inside_pts(pts, img_h, img_w):
    """Find bboxes as long as a part of bboxes is inside the image.

    Args:
        pts (Tensor): Shape (N, 2).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        Tensor: Index of the remaining pts.
    """
    inside_inds = (pts[:, 0] < img_w) & (pts[:, 0] > 0) \
                  & (pts[:, 1] < img_h) & (pts[:, 1] > 0)
    return inside_inds


def normalize(img: np.ndarray, mean, std):
    assert img.ndim == 3 and img.shape[-1] == 3
    assert img.dtype in [np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32")]

    mean = np.reshape(mean, (1, 1, -1))
    stdinv = 1. / np.reshape(std, (1, 1, -1))
    img = img.astype(np.float32)
    img -= mean
    img *= stdinv
    return img


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).astype(dtype)


def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(img.dtype)
            )
    return img.astype("float32") / max_value


def image_compression(img, quality, image_type):
    if image_type in [".jpeg", ".jpg"]:
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        raise NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        print(
            "Image compression augmentation "
            "is most effective with uint8 inputs, "
            "{} is used as input.".format(input_dtype),
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for image augmentation".format(input_dtype))

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    if needs_float:
        img = to_float(img, max_value=255)
    return img
