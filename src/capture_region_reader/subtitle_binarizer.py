import cv2
import numpy as np


def binarize_subtitle(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mean_val = float(np.mean(gray))
    if mean_val < 128:
        binary = cv2.bitwise_not(otsu)
    else:
        binary = otsu

    white_ratio = float(np.mean(binary == 255))
    if white_ratio < 0.4:
        binary = cv2.bitwise_not(binary)

    binary = _remove_noise_components(binary, h, w)

    result = np.full((h, w, 3), 255, dtype=np.uint8)
    result[binary == 0] = [0, 0, 0]

    return result


def _remove_noise_components(
    binary: np.ndarray, h: int, w: int,
) -> np.ndarray:
    text_mask = cv2.bitwise_not(binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_mask, connectivity=8,
    )

    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return binary

    total_area = h * w
    min_area = max(2, int(total_area * 0.00002))

    heights = stats[1:, cv2.CC_STAT_HEIGHT]

    valid_heights = heights[heights < h * 0.8]
    if len(valid_heights) >= 3:
        median_h = float(np.median(valid_heights))
    elif len(valid_heights) > 0:
        median_h = float(np.mean(valid_heights))
    else:
        median_h = h * 0.3

    cy_values = centroids[1:, 1]
    if len(cy_values) >= 3:
        text_y_center = float(np.median(cy_values))
        text_y_spread = max(median_h * 2, float(np.std(cy_values)) * 3)
    else:
        text_y_center = h / 2.0
        text_y_spread = h / 2.0

    result = np.full_like(binary, 255)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        cy = centroids[i, 1]

        if area < min_area:
            continue

        if comp_h > h * 0.85 or comp_w > w * 0.85:
            continue

        if area > total_area * 0.4:
            continue

        if median_h > 0 and comp_h > median_h * 6:
            continue

        if comp_h > 0 and comp_w / comp_h > 20:
            continue

        y_dist = abs(cy - text_y_center)
        if y_dist > text_y_spread and area < total_area * 0.005:
            continue

        result[labels == i] = 0

    return result
