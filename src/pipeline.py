"""Main vectorisation pipeline."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.vectorize.analyze import analyze_image
from src.vectorize.auto_params import auto_params
from src.vectorize.contour import build_paths
from src.vectorize.quantize import quantize_lab
from src.vectorize.svg_writer import write_svg

logger = logging.getLogger(__name__)


def _run_vectorize(
    img_rgb: np.ndarray,
    out_path: str,
    params: Dict[str, Any],
    stroke_mode: bool = False,
) -> str:
    """Core vectorisation step; returns *out_path*."""
    from src.preprocess.denoise import denoise  # noqa: PLC0415

    denoised = denoise(img_rgb, bilateral_iters=params["bilateral_iters"])
    labels, palette = quantize_lab(denoised, K=params["K"])

    h, w = img_rgb.shape[:2]

    # Sort colour indices by descending area (large regions go to the bottom)
    order = sorted(range(params["K"]), key=lambda i: -(labels == i).sum())

    layers = []
    for rank, i in enumerate(order):
        mask = (labels == i).astype(np.uint8) * 255
        path_d = build_paths(
            mask,
            morph=params["morph"],
            min_area=params["min_area"],
            smooth_eps=params["smooth_eps"],
        )
        if path_d is None:
            continue
        color_hex = "#%02x%02x%02x" % tuple(int(c) for c in palette[i])
        layers.append((color_hex, path_d))

    write_svg(out_path, w, h, layers, stroke_mode=stroke_mode)
    return out_path


def run(
    in_path: str,
    out_path: str,
    stroke_mode: bool = False,
    remove_bg: bool = False,
    remove_watermark: bool = False,
    industrial: bool = False,
    report: Optional[Dict[str, Any]] = None,
) -> str:
    """Run the full pipeline on a single image.

    Parameters
    ----------
    in_path:
        Input raster image path.
    out_path:
        Destination SVG path.
    stroke_mode:
        If True, output stroke-only (no fill).
    remove_bg:
        Apply rembg background removal pre-processing.
    remove_watermark:
        Apply FFT-based watermark removal pre-processing.
    industrial:
        Enable quality self-evaluation and auto-retry.
    report:
        If a dict is provided, pipeline metrics are written into it.

    Returns
    -------
    Path to the final SVG file.
    """
    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {in_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Pre-processing ────────────────────────────────────────────────────────
    if remove_bg:
        from src.preprocess.remove_bg import remove_background  # noqa: PLC0415

        result = remove_background(img_rgb)
        if result.shape[2] == 4:
            # Composite onto white background
            alpha = result[..., 3:4].astype(np.float32) / 255.0
            img_rgb = (result[..., :3].astype(np.float32) * alpha
                       + np.ones_like(result[..., :3], dtype=np.float32) * 255 * (1 - alpha)
                       ).astype(np.uint8)
        else:
            img_rgb = result

    if remove_watermark:
        from src.preprocess.remove_watermark import remove_watermark as _rw  # noqa: PLC0415

        img_rgb = _rw(img_rgb)

    # ── Feature analysis & adaptive parameters ────────────────────────────────
    feat = analyze_image(img_rgb)
    params = auto_params(feat)

    logger.info(
        "%s %dx%d uc=%.0f noise=%.1f → K=%d morph=%d min_area=%d eps=%.2f",
        Path(in_path).name,
        int(feat["w"]), int(feat["h"]),
        feat["unique_colors"], feat["noise"],
        params["K"], params["morph"], params["min_area"], params["smooth_eps"],
    )

    # ── Vectorisation (with optional quality retry) ───────────────────────────
    if industrial:
        from src.quality.auto_retry import run_with_retry  # noqa: PLC0415

        tmp_dir = tempfile.mkdtemp()

        def _fn(p: Dict[str, Any]) -> str:
            attempt_path = os.path.join(
                tmp_dir,
                f"attempt_K{p['K']}.svg",
            )
            return _run_vectorize(img_rgb, attempt_path, p, stroke_mode)

        final_svg, score, best_params = run_with_retry(
            _fn, params, img_rgb
        )
        # Copy the best result to the desired output path
        import shutil  # noqa: PLC0415

        shutil.copy2(final_svg, out_path)
        params = best_params
    else:
        score = None
        _run_vectorize(img_rgb, out_path, params, stroke_mode)

    # ── Record metrics ────────────────────────────────────────────────────────
    if report is not None:
        report[in_path] = {
            "out": out_path,
            "params": params,
            "ssim": score,
        }

    logger.info("→ %s  (SSIM=%s)", out_path, f"{score:.4f}" if score is not None else "n/a")
    return out_path
