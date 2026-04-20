"""Background removal using rembg (optional; falls back gracefully)."""
from __future__ import annotations

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


def remove_background(img_rgb: np.ndarray) -> np.ndarray:
    """Attempt to remove the background with rembg.

    Falls back to returning the original image if rembg is not installed or
    raises any exception.

    Parameters
    ----------
    img_rgb:
        Input image (H×W×3, uint8, RGB).

    Returns
    -------
    Image with background removed (H×W×4, uint8, RGBA) on success, or the
    original *img_rgb* unchanged on failure.
    """
    try:
        from PIL import Image
        import rembg  # noqa: PLC0415

        pil_img = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        result_bytes = rembg.remove(buf.read())
        result = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        return np.array(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rembg background removal failed (%s); using original image.", exc)
        return img_rgb
