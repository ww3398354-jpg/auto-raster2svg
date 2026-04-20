"""Command-line entry point for auto-raster2svg.

Usage examples
--------------
Single file::

    python -m src.cli input.png output.svg

Batch folder::

    python -m src.cli ./imgs ./svgs --workers 4

Stroke-only mode::

    python -m src.cli input.png output.svg --stroke

Industrial mode (quality retry)::

    python -m src.cli input.png output.svg --industrial

Skip optional pre-processing::

    python -m src.cli input.png output.svg --no-bg-remove --no-watermark
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.pipeline import run
from src.utils import collect_images, ensure_dir, stem_to_svg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _process_one(args_tuple: tuple) -> Dict[str, Any]:
    """Worker function for parallel processing."""
    in_path, out_path, kwargs = args_tuple
    report: Dict[str, Any] = {}
    try:
        run(str(in_path), str(out_path), report=report, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to process %s: %s", in_path, exc)
        report[str(in_path)] = {"error": str(exc)}
    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Auto raster-to-SVG vectorization pipeline",
    )
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("output", help="Output SVG file or directory")
    parser.add_argument("--stroke", action="store_true",
                        help="Output stroke-only contours (no fill)")
    parser.add_argument("--industrial", action="store_true",
                        help="Enable quality self-evaluation and auto-retry")
    parser.add_argument("--no-bg-remove", dest="remove_bg",
                        action="store_false", default=False,
                        help="Disable background removal (default: disabled)")
    parser.add_argument("--bg-remove", dest="remove_bg",
                        action="store_true",
                        help="Enable background removal (requires rembg)")
    parser.add_argument("--no-watermark", dest="remove_watermark",
                        action="store_false", default=False,
                        help="Disable watermark removal (default: disabled)")
    parser.add_argument("--watermark", dest="remove_watermark",
                        action="store_true",
                        help="Enable watermark removal")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (batch mode only)")
    parser.add_argument("--report", default="report.json",
                        help="Path for the output report JSON (default: report.json)")
    args = parser.parse_args(argv)

    run_kwargs = dict(
        stroke_mode=args.stroke,
        remove_bg=args.remove_bg,
        remove_watermark=args.remove_watermark,
        industrial=args.industrial,
    )

    in_path = Path(args.input)
    out_path = Path(args.output)

    # ── Single file mode ──────────────────────────────────────────────────────
    if in_path.is_file():
        report: Dict[str, Any] = {}
        run(str(in_path), str(out_path), report=report, **run_kwargs)
        _save_report(report, args.report)
        return 0

    # ── Batch / directory mode ────────────────────────────────────────────────
    files = collect_images(in_path)
    if not files:
        logger.error("No image files found in %s", in_path)
        return 1

    ensure_dir(out_path)
    tasks = [
        (f, stem_to_svg(f, out_path), run_kwargs)
        for f in files
    ]

    combined_report: Dict[str, Any] = {}

    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, t): t[0] for t in tasks}
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Vectorizing",
            ):
                combined_report.update(fut.result())
    else:
        for task in tqdm(tasks, desc="Vectorizing"):
            combined_report.update(_process_one(task))

    _save_report(combined_report, args.report)
    return 0


def _save_report(report: Dict[str, Any], path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        logger.info("Report saved → %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save report: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
