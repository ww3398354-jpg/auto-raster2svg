"""Utility helpers for file I/O."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def collect_images(src: str | Path) -> List[Path]:
    """Return all image files under *src* (directory) or the single file itself."""
    src = Path(src)
    if src.is_file():
        return [src]
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(src.glob(f"*{ext}"))
        files.extend(src.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stem_to_svg(src: Path, dst_dir: Path) -> Path:
    """Map an input image path to an output SVG path inside *dst_dir*."""
    return dst_dir / (src.stem + ".svg")
