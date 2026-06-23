from __future__ import annotations

import cv2


WINDOW_NAME = "Face Blur Preview"


def show_preview(frame) -> bool:
    """Render preview frame and return False when user presses Q."""
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"))


def close_preview() -> None:
    cv2.destroyAllWindows()