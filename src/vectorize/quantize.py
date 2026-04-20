"""Colour quantisation in CIELAB space using MiniBatchKMeans."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def quantize_lab(img_rgb: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise *img_rgb* into *K* colours using CIELAB MiniBatchKMeans.

    Parameters
    ----------
    img_rgb:
        Input image (H×W×3, uint8, RGB).
    K:
        Number of colour clusters.

    Returns
    -------
    labels:
        Integer label map (H×W, int32), values in [0, K).
    palette:
        Cluster centre colours (K×3, uint8, RGB).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3).astype(np.float32)

    km = MiniBatchKMeans(
        n_clusters=K,
        n_init=4,
        random_state=0,
        batch_size=4096,
        max_iter=100,
    ).fit(pixels)

    labels = km.labels_.reshape(h, w).astype(np.int32)

    # Convert cluster centres from LAB → RGB
    centers_lab = km.cluster_centers_.astype(np.uint8).reshape(1, -1, 3)
    palette = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)

    return labels, palette
