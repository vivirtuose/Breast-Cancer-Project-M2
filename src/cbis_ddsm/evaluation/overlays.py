"""OpenCV-only qualitative overlays: image plus ground-truth / prediction contour panels."""

from __future__ import annotations

import cv2
import numpy as np

_GREEN_BGR = (0, 200, 0)
_RED_BGR = (0, 0, 220)
_BAND_HEIGHT = 28


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a HxW grayscale image (float in [0, 1] or uint8) to a BGR uint8 tile."""
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)


def _draw_contours(tile: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Draw the external contours of a boolean mask onto `tile` in-place; returns `tile`."""
    binary = np.asarray(mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tile, contours, -1, color, 1)
    return tile


def make_overlay(image: np.ndarray, mask: np.ndarray, prediction: np.ndarray, title: str) -> np.ndarray:
    """Build a three-tile BGR panel: image, image+ground-truth contour (green), image+prediction contour (red).

    Tiles keep the input HxW size (no rescaling) and are placed side by side; a title band is
    drawn above them. `image` may be float in [0, 1] or uint8; `mask`/`prediction` are boolean
    HxW arrays. Returns a BGR uint8 array of shape `(size + band, 3 * size, 3)`.
    """
    base = _to_bgr_uint8(image)

    tile_image = base.copy()
    tile_truth = _draw_contours(base.copy(), mask, _GREEN_BGR)
    tile_pred = _draw_contours(base.copy(), prediction, _RED_BGR)
    tiles = np.concatenate([tile_image, tile_truth, tile_pred], axis=1)

    band = np.zeros((_BAND_HEIGHT, tiles.shape[1], 3), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = max(0, (band.shape[1] - text_width) // 2)
    text_y = (_BAND_HEIGHT + text_height) // 2
    cv2.putText(band, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return np.concatenate([band, tiles], axis=0)
