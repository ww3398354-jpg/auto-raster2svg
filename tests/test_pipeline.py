"""Pipeline integration tests using synthetically generated images."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import run


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tricolor_image() -> np.ndarray:
    """Create a 256×256 synthetic three-colour image (no external files needed)."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Red region (top-left quadrant)
    img[:128, :128] = (220, 50, 50)
    # Green region (top-right quadrant)
    img[:128, 128:] = (50, 200, 50)
    # Blue region (bottom half)
    img[128:, :] = (50, 50, 200)
    return img


def _save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    Image.fromarray(img).save(str(path))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_svg_created(self, tmp_path: Path) -> None:
        """pipeline.run() must produce an SVG file."""
        in_png = tmp_path / "input.png"
        out_svg = tmp_path / "output.svg"
        _save_png(_make_tricolor_image(), in_png)

        run(str(in_png), str(out_svg), remove_bg=False, remove_watermark=False)

        assert out_svg.exists(), "SVG file was not created"
        assert out_svg.stat().st_size > 0, "SVG file is empty"

    def test_svg_valid_xml(self, tmp_path: Path) -> None:
        """The output SVG must be well-formed XML."""
        in_png = tmp_path / "input.png"
        out_svg = tmp_path / "output.svg"
        _save_png(_make_tricolor_image(), in_png)

        run(str(in_png), str(out_svg), remove_bg=False, remove_watermark=False)

        tree = ET.parse(str(out_svg))
        root = tree.getroot()
        assert root is not None

    def test_svg_has_groups(self, tmp_path: Path) -> None:
        """SVG must contain ≥ 2 <g> elements (colour layers)."""
        in_png = tmp_path / "input.png"
        out_svg = tmp_path / "output.svg"
        _save_png(_make_tricolor_image(), in_png)

        run(str(in_png), str(out_svg), remove_bg=False, remove_watermark=False)

        tree = ET.parse(str(out_svg))
        # Try with and without namespace
        groups = tree.findall(".//{http://www.w3.org/2000/svg}g")
        if not groups:
            groups = tree.findall(".//g")
        assert len(groups) >= 2, f"Expected ≥2 <g> elements, got {len(groups)}"

    def test_svg_has_paths(self, tmp_path: Path) -> None:
        """SVG must contain ≥ 1 <path> element."""
        in_png = tmp_path / "input.png"
        out_svg = tmp_path / "output.svg"
        _save_png(_make_tricolor_image(), in_png)

        run(str(in_png), str(out_svg), remove_bg=False, remove_watermark=False)

        tree = ET.parse(str(out_svg))
        paths = tree.findall(".//{http://www.w3.org/2000/svg}path")
        if not paths:
            paths = tree.findall(".//path")
        assert len(paths) >= 1, f"Expected ≥1 <path> element, got {len(paths)}"

    def test_stroke_mode(self, tmp_path: Path) -> None:
        """Stroke mode must produce an SVG with fill='none'."""
        in_png = tmp_path / "input.png"
        out_svg = tmp_path / "output.svg"
        _save_png(_make_tricolor_image(), in_png)

        run(str(in_png), str(out_svg),
            stroke_mode=True, remove_bg=False, remove_watermark=False)

        content = out_svg.read_text()
        assert 'fill="none"' in content, "Stroke mode should produce fill='none'"
