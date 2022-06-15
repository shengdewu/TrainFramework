import torch


def adjust_brightness_adaptive(img: torch.Tensor, min_brightness_factor: float, max_brightness_factor: float, max_brightness_stretch: float = 1.0) -> torch.Tensor:
    """
    when brightening, the darker is brighter, when press dark, the brighter is darker
    """
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    enhance_img = torch.zeros_like(img)

    k = (min_brightness_factor - max_brightness_factor) / (max_brightness_stretch-0.0)
    b = min_brightness_factor - max_brightness_stretch * k
    exposure = gray * k + b

    enhance_img[0] = exposure * img[0]
    enhance_img[1] = exposure * img[1]
    enhance_img[2] = exposure * img[2]
    return enhance_img


def adjust_brightness_adaptive_inv(img: torch.Tensor, min_brightness_factor: float, max_brightness_factor: float, max_brightness_stretch: float = 1.0) -> torch.Tensor:
    """
    when brightening the brighter is brighter, when press dark, the darker is darker,
    """
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    enhance_img = torch.zeros_like(img)

    k = (max_brightness_factor - min_brightness_factor) / (max_brightness_stretch-0.0)
    b = max_brightness_factor - max_brightness_stretch * k
    exposure = gray * k + b

    enhance_img[0] = exposure * img[0]
    enhance_img[1] = exposure * img[1]
    enhance_img[2] = exposure * img[2]
    return enhance_img
