from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_input_video(path: str) -> Path:
    input_path = Path(path)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported input format '{input_path.suffix}'. Supported formats: {allowed}"
        )

    return input_path


def validate_input_image(path: str) -> Path:
    input_path = Path(path)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported input format '{input_path.suffix}'. Supported formats: {allowed}"
        )

    return input_path


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_blurred.mp4")


def default_image_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_blurred{input_path.suffix}")


def create_progress_bar(total_frames: int):
    if total_frames > 0:
        return tqdm(total=total_frames, desc="Processing frames", unit="frame")
    return tqdm(desc="Processing frames", unit="frame")


def print_config_summary(
    input_path: Path,
    output_path: Path,
    blur_strength: int,
    detection_interval: int,
    preview_enabled: bool,
) -> None:
    print("\nConfiguration")
    print(f"- Input: {input_path}")
    print(f"- Output: {output_path}")
    print(f"- Blur strength: {blur_strength}")
    print(f"- Detection interval: {detection_interval}")
    print(f"- Preview: {'enabled' if preview_enabled else 'disabled'}\n")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def mux_audio_with_ffmpeg(input_video: Path, processed_video: Path, output_video: Path) -> bool:
    """Mux original audio into processed video. Returns True if audio was muxed."""
    if not ffmpeg_available():
        print("Warning: ffmpeg not found. Saving video without original audio.")
        shutil.copy2(processed_video, output_video)
        return False

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(processed_video),
        "-i",
        str(input_video),
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print("Warning: ffmpeg mux failed. Saving processed video without audio.")
        if exc.stderr:
            print(exc.stderr.strip())
        shutil.copy2(processed_video, output_video)
        return False
