import cv2
import numpy as np
import numba as nb


def polar2cart(img, pad: int = 250, scale=1.0):
    """
    Polar (linear) to cartesian (circular) image.
    pad: padding at the top to compensate for probe radius
    scale: default 1.0. Use smaller to get smaller image output
    """
    img = cv2.copyMakeBorder(img, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    h, w = img.shape[:2]
    r = round(min(h, w) * scale)
    sz = r * 2
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS
    img = cv2.warpPolar(img, (sz, sz), (r, r), r, flags)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def cart2polar(img: np.ndarray, target_sz: tuple[int, int], pad: int = 250):
    """
    img: a square image, circular
    target_sz: [x, y]
    pad: padding used for polar2cart
    """
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS
    assert img.shape[0] == img.shape[1]
    r = img.shape[0] // 2
    img = cv2.warpPolar(img, (pad + target_sz[1], target_sz[0]), (r, r), r, flags)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img[pad:]
    return img


@nb.njit(nogil=True, fastmath=True, parallel=True, cache=True)
def log_compress_par(x, dB: float):
    "Log compression with dynamic range dB"
    maxval = 255.0
    res = np.empty(x.shape, dtype=np.uint8)
    l = len(x)
    for i in nb.prange(l):
        xmax = np.percentile(x[i], 99.9)
        lc = 20.0 / dB * np.log10(x[i] / xmax) + 1.0
        lc = np.clip(lc, 0.0, 1.0)
        res[i] = (maxval * lc).astype(np.uint8)
    return res
