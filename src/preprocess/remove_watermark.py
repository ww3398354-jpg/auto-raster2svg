"""Watermark removal via FFT periodic-noise detection + inpainting."""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FFT_PEAK_THRESHOLD = 10.0  # times the median — conservative


def _detect_periodic_mask(gray: np.ndarray, radius: int = 3) -> np.ndarray:
    """Return a binary mask of periodic FFT peaks (potential watermark pattern)."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Normalise to uint8 for thresholding
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Remove the DC component (centre)
    h, w = norm.shape
    cy, cx = h // 2, w // 2
    cv2.circle(norm, (cx, cy), max(h, w) // 8, 0, -1)

    median_val = float(np.median(norm[norm > 0])) if norm.any() else 1.0
    threshold = min(median_val * _FFT_PEAK_THRESHOLD, 254)
    _, mask_fft = cv2.threshold(norm, threshold, 255, cv2.THRESH_BINARY)

    # Dilate slightly so inpainting covers the ringing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask_fft = cv2.dilate(mask_fft, kernel)

    # Map FFT mask back to image space (same size by construction of fftshift)
    if mask_fft.shape != gray.shape:
        mask_fft = cv2.resize(mask_fft, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask_fft


def remove_watermark(img_rgb: np.ndarray) -> np.ndarray:
    """Detect and remove periodic watermark patterns.

    Uses FFT to identify periodic peaks in the frequency domain, builds a
    spatial inpainting mask, and applies ``cv2.INPAINT_TELEA``.  If no
    significant periodic pattern is detected the original image is returned
    unchanged.

    Parameters
    ----------
    img_rgb:
        Input image (H×W×3, uint8, RGB).

    Returns
    -------
    Cleaned image (H×W×3, uint8, RGB).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = _detect_periodic_mask(gray)

    if mask.sum() == 0:
        logger.debug("No periodic watermark pattern detected; skipping inpainting.")
        return img_rgb

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
