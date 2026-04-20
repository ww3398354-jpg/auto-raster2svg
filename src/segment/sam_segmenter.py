"""SAM2 automatic segmenter — lazy-loaded optional extra."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_sam2_available: Optional[bool] = None


def _check_sam2() -> bool:
    global _sam2_available
    if _sam2_available is None:
        try:
            import segment_anything  # noqa: F401
            _sam2_available = True
        except ImportError:
            try:
                import sam2  # noqa: F401
                _sam2_available = True
            except ImportError:
                _sam2_available = False
    return _sam2_available


class SAMSegmenter:
    """Thin wrapper around SAM2 automatic mask generator.

    Falls back gracefully when SAM2 is not installed.
    """

    def __init__(self, model_type: str = "vit_b", checkpoint: Optional[str] = None) -> None:
        self._model_type = model_type
        self._checkpoint = checkpoint
        self._generator = None

    def _load(self) -> bool:
        """Lazy-load SAM2.  Returns True on success."""
        if self._generator is not None:
            return True
        if not _check_sam2():
            logger.warning(
                "SAM2 / segment-anything is not installed. "
                "Install it as an extra: pip install segment-anything"
            )
            return False
        try:
            from segment_anything import (  # type: ignore[import-not-found]
                SamAutomaticMaskGenerator,
                sam_model_registry,
            )

            if self._checkpoint is None:
                logger.warning("SAM2 checkpoint path not provided; skipping segmentation.")
                return False
            sam = sam_model_registry[self._model_type](checkpoint=self._checkpoint)
            self._generator = SamAutomaticMaskGenerator(sam)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load SAM2 model: %s", exc)
            return False

    def segment(self, img_rgb: np.ndarray) -> List[np.ndarray]:
        """Return a list of binary masks (H×W, bool) for each detected region.

        Returns an empty list when SAM2 is unavailable or fails.
        """
        if not self._load():
            return []
        try:
            masks_data = self._generator.generate(img_rgb)
            return [m["segmentation"] for m in masks_data]
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAM2 segmentation failed: %s", exc)
            return []
