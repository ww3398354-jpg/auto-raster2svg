"""Write layered SVG files with one <g> element per colour."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import svgwrite


def write_svg(
    out_path: str,
    width: int,
    height: int,
    layers: List[Tuple[str, str]],
    stroke_mode: bool = False,
) -> None:
    """Write a layered SVG file.

    Parameters
    ----------
    out_path:
        Destination file path.
    width, height:
        Canvas dimensions in pixels.
    layers:
        List of ``(color_hex, path_d)`` tuples, ordered bottom-to-top.
    stroke_mode:
        If True, render strokes only (no fill).
    """
    dwg = svgwrite.Drawing(out_path, size=(width, height), viewBox=f"0 0 {width} {height}")

    for rank, (color_hex, path_d) in enumerate(layers):
        group_id = f"L{rank:02d}_{color_hex.lstrip('#')}"
        if stroke_mode:
            g = dwg.g(
                id=group_id,
                fill="none",
                stroke=color_hex,
                stroke_width="1",
            )
        else:
            g = dwg.g(
                id=group_id,
                fill=color_hex,
                stroke="none",
                **{"fill-rule": "evenodd"},
            )
        g.add(dwg.path(d=path_d))
        dwg.add(g)

    dwg.save()
