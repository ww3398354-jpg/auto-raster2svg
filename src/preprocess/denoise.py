"""Edge-preserving denoising: bilateral filter + recursive filter."""
from __future__ import annotations

import cv2
import numpy as np


def denoise(img_rgb: np.ndarray, bilateral_iters: int = 2) -> np.ndarray:
    """Denoise *img_rgb* while preserving edges.

    Applies *bilateral_iters* rounds of ``cv2.bilateralFilter`` followed by
    one pass of ``cv2.edgePreservingFilter`` with ``RECURS_FILTER``.

    Parameters
    ----------
    img_rgb:
        Input image (H×W×3, uint8, RGB).
    bilateral_iters:
        Number of bilateral filter iterations (1–3 recommended).

    Returns
    -------
    Denoised image (H×W×3, uint8, RGB).
    """
    out = img_rgb.copy()
    for _ in range(bilateral_iters):
        out = cv2.bilateralFilter(out, d=9, sigmaColor=60, sigmaSpace=60)
    out = cv2.edgePreservingFilter(
        out,
        flags=cv2.RECURS_FILTER,
        sigma_s=40,
        sigma_r=0.35,
    )
    return out
