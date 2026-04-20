"""Adaptive parameter selection based on image features."""
from __future__ import annotations

import math
from typing import Any, Dict


def auto_params(feat: Dict[str, Any]) -> Dict[str, Any]:
    """Derive algorithm parameters from image feature metrics.

    Parameters
    ----------
    feat:
        Dict returned by :func:`src.vectorize.analyze.analyze_image`.

    Returns
    -------
    dict with keys:
        K               – number of colour clusters
        bilateral_iters – bilateral filter iterations
        morph           – morphology kernel size (pixels)
        min_area        – minimum contour area (pixels)
        smooth_eps      – contour approximation epsilon
    """
    h = int(feat["h"])
    w = int(feat["w"])
    area = h * w

    # ── Colour count K ────────────────────────────────────────────────────────
    uc = feat["unique_colors"]
    if uc < 500:
        K = 3
    elif uc < 2000:
        K = 5
    elif uc < 8000:
        K = 8
    elif uc < 20000:
        K = 12
    else:
        K = 16

    # ── Bilateral filter iterations ───────────────────────────────────────────
    noise = feat["noise"]
    if noise < 50:
        bilateral_iters = 1
    elif noise < 200:
        bilateral_iters = 2
    else:
        bilateral_iters = 3

    # ── Morphology kernel (resolution-adaptive) ───────────────────────────────
    morph = max(2, round(math.sqrt(area) / 400))

    # ── Minimum contour area ──────────────────────────────────────────────────
    min_area = max(20, int(area * 0.00015))

    # ── Contour smoothing epsilon ─────────────────────────────────────────────
    smooth_eps = max(0.8, math.sqrt(area) / 1200)

    return dict(
        K=K,
        bilateral_iters=bilateral_iters,
        morph=morph,
        min_area=min_area,
        smooth_eps=smooth_eps,
    )
