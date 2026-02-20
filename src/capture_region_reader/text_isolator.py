"""Isolate subtitle text from a screenshot before OCR.

Delegates to the 3-step pipeline from subtitle_detector/binarizer/cleaner:
1. detect_and_crop() finds text characters, groups into blocks, refines bounds
2. binarize_subtitle() converts to clean black-on-white
3. clean_artifacts() removes logos, timestamps, non-text elements
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from capture_region_reader.subtitle_detector import detect_and_crop

logger = logging.getLogger(__name__)


def isolate_text(rgb: np.ndarray) -> np.ndarray | None:
    """Isolate subtitle text from a screenshot.

    Returns a clean black-text-on-white RGB image, or None if no text found.

    Converts RGB input to BGR for the detection pipeline, then converts
    the result back to RGB.

    Args:
        rgb: Input image in RGB format (H x W x 3).
    """
    h, w = rgb.shape[:2]
    if h < 10 or w < 10:
        return None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result = detect_and_crop(bgr)

    if result is None:
        logger.debug("detect_and_crop returned None (input %dx%d)", w, h)
        return None

    logger.debug(
        "detect_and_crop OK: input %dx%d -> output %dx%d",
        w, h, result.shape[1], result.shape[0],
    )
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
