"""SSIM-based quality evaluator: render SVG → PNG → compare with original."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _render_svg(svg_path: str, width: int, height: int) -> np.ndarray:
    """Render *svg_path* to an RGB numpy array via cairosvg."""
    import cairosvg  # noqa: PLC0415

    png_bytes = cairosvg.svg2png(
        url=svg_path,
        output_width=width,
        output_height=height,
    )
    from PIL import Image  # noqa: PLC0415
    import io  # noqa: PLC0415

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.array(img)


def compute_ssim(original_rgb: np.ndarray, svg_path: str) -> float:
    """Compute SSIM between *original_rgb* and the rendered *svg_path*.

    Returns a float in [0, 1].  Returns 0.0 on any failure.
    """
    try:
        from skimage.metrics import structural_similarity  # noqa: PLC0415

        h, w = original_rgb.shape[:2]
        rendered = _render_svg(svg_path, w, h)

        # Resize rendered image to match original in case of rounding
        if rendered.shape[:2] != (h, w):
            rendered = cv2.resize(rendered, (w, h))

        score, _ = structural_similarity(
            original_rgb, rendered, full=True, channel_axis=2
        )
        return float(score)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SSIM evaluation failed: %s", exc)
        return 0.0
