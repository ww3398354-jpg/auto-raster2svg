"""Auto-retry logic: re-run vectorisation when SSIM is below threshold."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from src.quality.evaluator import compute_ssim

logger = logging.getLogger(__name__)

SSIM_THRESHOLD = 0.82
MAX_RETRIES = 3


def run_with_retry(
    run_fn: Callable[[Dict[str, Any]], str],
    initial_params: Dict[str, Any],
    original_rgb: Any,
    ssim_threshold: float = SSIM_THRESHOLD,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, float, Dict[str, Any]]:
    """Run *run_fn* with quality-gated auto-retry.

    Parameters
    ----------
    run_fn:
        Callable that accepts a params dict and returns the path to the
        generated SVG file.
    initial_params:
        Starting parameters (must contain at least ``K`` and ``smooth_eps``).
    original_rgb:
        Original image array used for SSIM comparison.
    ssim_threshold:
        Minimum acceptable SSIM score.
    max_retries:
        Maximum number of additional attempts after the first run.

    Returns
    -------
    (best_svg_path, best_score, best_params)
    """
    params = dict(initial_params)
    best_svg: Optional[str] = None
    best_score: float = -1.0
    best_params: Dict[str, Any] = dict(params)

    for attempt in range(max_retries + 1):
        svg_path = run_fn(params)
        score = compute_ssim(original_rgb, svg_path)

        logger.info("Attempt %d/%d — SSIM=%.4f (K=%d, eps=%.2f)",
                    attempt + 1, max_retries + 1, score,
                    params.get("K", "?"), params.get("smooth_eps", "?"))

        if score > best_score:
            best_score = score
            best_svg = svg_path
            best_params = dict(params)

        if score >= ssim_threshold:
            break

        if attempt < max_retries:
            # Escalate: more colours, tighter contours
            params = dict(params)
            params["K"] = int(params.get("K", 5) * 1.5)
            params["smooth_eps"] = params.get("smooth_eps", 1.0) * 0.7
            logger.info("Quality below threshold (%.2f < %.2f); retrying with K=%d.",
                        score, ssim_threshold, params["K"])

    return best_svg, best_score, best_params
