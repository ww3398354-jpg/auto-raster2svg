"""Contour extraction and Catmull-Rom → cubic Bézier conversion."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def _catmull_rom_to_bezier(pts: np.ndarray) -> str:
    """Convert a closed polygon to a cubic Bézier SVG path using Catmull-Rom.

    Control points:  c1 = p1 + (p2 - p0) / 6
                     c2 = p2 - (p3 - p1) / 6
    """
    n = len(pts)
    parts = [f"M {pts[0, 0]:.2f},{pts[0, 1]:.2f}"]
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        parts.append(
            f"C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}"
        )
    parts.append("Z")
    return " ".join(parts)


def _contour_to_path(contour: np.ndarray, smooth_eps: float) -> Optional[str]:
    """Simplify *contour* and convert it to an SVG path *d* string."""
    simplified = cv2.approxPolyDP(contour, smooth_eps, True).squeeze()
    if simplified.ndim != 2 or len(simplified) < 3:
        return None
    return _catmull_rom_to_bezier(simplified.astype(float))


def build_paths(
    mask: np.ndarray,
    morph: int,
    min_area: int,
    smooth_eps: float,
) -> Optional[str]:
    """Apply morphology, extract contours, and return a combined SVG path string.

    Parameters
    ----------
    mask:
        Binary mask (H×W, uint8, values 0 or 255).
    morph:
        Morphology kernel size.
    min_area:
        Contours smaller than this area (pixels) are discarded.
    smooth_eps:
        ``approxPolyDP`` epsilon for contour simplification.

    Returns
    -------
    SVG *d* attribute string combining all qualifying contours, or ``None``
    if no valid contours were found.
    """
    kernel = np.ones((morph, morph), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )

    parts: List[str] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        d = _contour_to_path(cnt, smooth_eps)
        if d:
            parts.append(d)

    return " ".join(parts) if parts else None
