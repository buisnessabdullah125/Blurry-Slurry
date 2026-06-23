from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


def normalize_blur_strength(value: int) -> int:
    """Ensure the mosaic block size is a valid odd integer >= 3."""
    if value < 3:
        value = 3
    if value % 2 == 0:
        value += 1
    return value


def blur_faces(frame, bboxes: List[BBox], blur_strength: int, padding_ratio: float = 0.15):
    """Apply oval mosaic masks to each face bounding box on a frame."""
    # Use a smaller block size than the raw strength so the mosaic stays detailed.
    block_size = max(4, normalize_blur_strength(blur_strength) // 6)
    h, w = frame.shape[:2]

    for x, y, bw, bh in bboxes:
        pad_x = max(2, int(bw * padding_ratio))
        pad_y = max(2, int(bh * padding_ratio))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]

        # Create a pixelated mosaic version of the ROI by downsampling and upsampling.
        small_w = max(1, (x2 - x1) // block_size)
        small_h = max(1, (y2 - y1) // block_size)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        center = ((x2 - x1) // 2, (y2 - y1) // 2)
        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        mask_3ch = cv2.merge([mask, mask, mask])
        inverse_mask = cv2.bitwise_not(mask_3ch)
        preserved = cv2.bitwise_and(roi, inverse_mask)
        oval_mosaic = cv2.bitwise_and(mosaic, mask_3ch)
        frame[y1:y2, x1:x2] = cv2.add(preserved, oval_mosaic)

    return frame
