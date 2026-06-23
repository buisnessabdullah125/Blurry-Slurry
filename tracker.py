from __future__ import annotations

from typing import List, Tuple

import cv2

BBox = Tuple[int, int, int, int]


def _try_create_tracker(name: str):
    creator_name = f"Tracker{name}_create"

    if hasattr(cv2, creator_name):
        return getattr(cv2, creator_name)()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, creator_name):
        return getattr(cv2.legacy, creator_name)()

    return None


def create_preferred_tracker():
    """Create best available OpenCV tracker, preferring CSRT."""
    for tracker_name in ("CSRT", "KCF", "MOSSE", "MIL"):
        tracker = _try_create_tracker(tracker_name)
        if tracker is not None:
            return tracker_name, tracker

    raise RuntimeError(
        "No supported OpenCV trackers are available in this build. "
        "Install 'opencv-contrib-python' to enable CSRT/KCF trackers."
    )


class FaceTrackerManager:
    def __init__(self, detection_interval: int = 15, max_missing_frames: int = 2) -> None:
        self.detection_interval = detection_interval
        self.max_missing_frames = max_missing_frames
        self.trackers = []
        self._printed_fallback_warning = False

    def initialize(self, frame, bboxes: List[BBox]) -> None:
        self.trackers = []
        for bbox in bboxes:
            tracker_name, tracker = create_preferred_tracker()
            if tracker_name != "CSRT" and not self._printed_fallback_warning:
                print(
                    f"Warning: CSRT unavailable in this OpenCV build. "
                    f"Falling back to {tracker_name} tracker."
                )
                self._printed_fallback_warning = True
            tracker.init(frame, bbox)
            self.trackers.append({"tracker": tracker, "last_box": bbox, "missing": 0})

    def update(self, frame) -> List[BBox]:
        tracked: List[BBox] = []
        alive_trackers = []

        for state in self.trackers:
            tracker = state["tracker"]
            ok, box = tracker.update(frame)
            if not ok:
                state["missing"] += 1
                if state["missing"] <= self.max_missing_frames:
                    tracked.append(state["last_box"])
                    alive_trackers.append(state)
                continue

            x, y, w, h = box
            xi, yi, wi, hi = int(x), int(y), int(w), int(h)
            if wi > 0 and hi > 0:
                current_box = (xi, yi, wi, hi)
                state["last_box"] = current_box
                state["missing"] = 0
                tracked.append(current_box)
                alive_trackers.append(state)

        self.trackers = alive_trackers
        return tracked