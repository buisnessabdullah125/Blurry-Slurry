from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from blurrer import blur_faces, normalize_blur_strength
from detector import FaceDetector
from preview import close_preview, show_preview
from tracker import FaceTrackerManager
from utils import (
    create_progress_bar,
    default_output_path,
    default_image_output_path,
    SUPPORTED_IMAGE_EXTENSIONS,
    mux_audio_with_ffmpeg,
    print_config_summary,
    validate_input_image,
    validate_input_video,
)

ProgressCallback = Callable[[int, int], None]


@dataclass
class VideoProcessResult:
    output_path: Path
    cancelled: bool
    audio_muxed: bool
    elapsed_seconds: float


@dataclass
class ImageProcessResult:
    output_path: Path
    face_count: int
    elapsed_seconds: float


def _blur_frame(frame, detector: FaceDetector, blur_strength: int, padding_ratio: float):
    detections = detector.detect(frame)
    processed = blur_faces(frame, detections, blur_strength, padding_ratio=padding_ratio)
    return processed, detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatically detect and blur faces in a video using YOLOv8 and OpenCV."
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output video file (default: <input>_blurred.mp4)",
    )
    parser.add_argument(
        "--blur-strength",
        type=int,
        default=51,
        help="Gaussian blur kernel size (must be odd)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a real-time preview window. Press Q to cancel.",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=15,
        help="Frames between YOLO re-detection passes",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.15,
        help="Extra face padding relative to the detected box size",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.blur_strength < 3:
        parser.error("--blur-strength must be >= 3 and odd.")
    if args.blur_strength % 2 == 0:
        parser.error("--blur-strength must be odd.")
    if args.detection_interval < 1:
        parser.error("--detection-interval must be >= 1.")
    if args.padding_ratio < 0:
        parser.error("--padding-ratio must be >= 0.")


def process_video_file(
    input_file: str | Path,
    output_file: str | Path | None = None,
    *,
    blur_strength: int = 51,
    detection_interval: int = 15,
    padding_ratio: float = 0.15,
    preview: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> VideoProcessResult:
    input_path = validate_input_video(str(input_file))
    output_path = Path(output_file) if output_file else default_output_path(input_path)
    blur_strength = normalize_blur_strength(blur_strength)

    print_config_summary(
        input_path=input_path,
        output_path=output_path,
        blur_strength=blur_strength,
        detection_interval=detection_interval,
        preview_enabled=preview,
    )

    start_time = time.time()
    detector = FaceDetector()
    tracker = FaceTrackerManager(detection_interval=detection_interval)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temporary file holds the processed video stream before optional audio mux.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        temp_video_path = Path(tmp_file.name)

    writer = cv2.VideoWriter(
        str(temp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Failed to initialize output video writer.")

    progress = create_progress_bar(total_frames)

    cancelled = False
    frame_index = 0

    try:
        ret, frame = capture.read()
        if not ret:
            raise RuntimeError("Could not read first frame from video.")

        processed, detections = _blur_frame(frame, detector, blur_strength, padding_ratio)
        tracker.initialize(frame, detections)
        writer.write(processed)
        progress.update(1)
        frame_index += 1
        if progress_callback:
            progress_callback(frame_index, total_frames)

        if preview and not show_preview(processed):
            cancelled = True

        while not cancelled:
            ret, frame = capture.read()
            if not ret:
                break

            if frame_index % detection_interval == 0:
                processed, boxes = _blur_frame(frame, detector, blur_strength, padding_ratio)
                tracker.initialize(frame, boxes)
            else:
                boxes = tracker.update(frame)
                processed = blur_faces(frame, boxes, blur_strength, padding_ratio=padding_ratio)
            writer.write(processed)

            if preview and not show_preview(processed):
                cancelled = True

            progress.update(1)
            frame_index += 1
            if progress_callback:
                progress_callback(frame_index, total_frames)

    finally:
        progress.close()
        capture.release()
        writer.release()
        close_preview()

    audio_muxed = mux_audio_with_ffmpeg(input_path, temp_video_path, output_path)

    try:
        temp_video_path.unlink(missing_ok=True)
    except OSError:
        pass

    elapsed = time.time() - start_time
    state = "cancelled by user" if cancelled else "completed"
    audio_state = "with original audio" if audio_muxed else "without original audio"

    print(f"\nProcessing {state}.")
    print(f"Output saved to: {output_path}")
    print(f"Audio: {audio_state}")
    print(f"Total processing time: {elapsed:.2f}s")

    return VideoProcessResult(
        output_path=output_path,
        cancelled=cancelled,
        audio_muxed=audio_muxed,
        elapsed_seconds=elapsed,
    )


def process_image_file(
    input_file: str | Path,
    output_file: str | Path | None = None,
    *,
    blur_strength: int = 51,
    padding_ratio: float = 0.15,
) -> ImageProcessResult:
    input_path = validate_input_image(str(input_file))
    output_path = Path(output_file) if output_file else default_image_output_path(input_path)
    if output_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        output_path = output_path.with_suffix(input_path.suffix)
    blur_strength = normalize_blur_strength(blur_strength)

    print_config_summary(
        input_path=input_path,
        output_path=output_path,
        blur_strength=blur_strength,
        detection_interval=1,
        preview_enabled=False,
    )

    start_time = time.time()
    detector = FaceDetector()
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f"Could not read image file: {input_path}")

    processed, detections = _blur_frame(frame, detector, blur_strength, padding_ratio)
    if not cv2.imwrite(str(output_path), processed):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    elapsed = time.time() - start_time
    print(f"\nProcessing completed.")
    print(f"Output saved to: {output_path}")
    print(f"Total processing time: {elapsed:.2f}s")

    return ImageProcessResult(
        output_path=output_path,
        face_count=len(detections),
        elapsed_seconds=elapsed,
    )


def process_video(args: argparse.Namespace) -> int:
    process_video_file(
        args.input,
        args.output,
        blur_strength=args.blur_strength,
        detection_interval=args.detection_interval,
        padding_ratio=args.padding_ratio,
        preview=args.preview,
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    try:
        return process_video(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
