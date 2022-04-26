""" Module containing function for upgrade  of RGB image """

import cv2
import numpy as np


def white_balance(img: np.ndarray) -> np.ndarray:
    r"""Upgrade white balance of RGB image.
    More about algorithm : https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html

    Args:
        img: np.ndarray [HxWx3], where 3 - number of color canals
    Returns:
        np.ndarray [HxWx3] : same image with upgraded white balance.
    Examples:
        >>> import cv2
        >>> img_path = 'img.png'
        >>> img_before = cv2.imread(img_path)
        >>> img_after  = white_balance(img_before)
        >>> final = np.hstack((img_before, img_after))
        >>> def show(photo):
        ...    cv2.imshow('Cool', photo)
        ...    cv2.waitKey(0)
        ...    cv2.destroyAllWindows()
        ...
        >>> show(final)
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
