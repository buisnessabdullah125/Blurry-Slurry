# Blurry-Slurry

An ai tool that detects faces in videos, tracks them between frames, and applies a configurable blur before exporting the processed output.

## What it does

- Detects faces with YOLOv8 using `yolov8n-face.pt`
- Tracks faces between detections to reduce the cost of repeated inference
- Blurs each detected face region with a configurable strength
- Re-detects faces every configurable number of frames
- Preserves the original audio when `ffmpeg` is available
- Can show a live preview window while processing

## Project Layout

- `main.py`: CLI entry point and video processing pipeline
- `detector.py`: YOLOv8 face detection wrapper with Haar fallback
- `tracker.py`: OpenCV tracker manager
- `blurrer.py`: Face blur and masking logic
- `preview.py`: OpenCV preview helpers
- `utils.py`: Validation, output naming, progress, and audio mux helpers

## Requirements

Install Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want audio preserved in the output, install `ffmpeg` and make sure it is available on your `PATH`.

## Usage

Process a video:

```bash
python main.py --input input.mp4 --output output.mp4
```

Useful options:

```bash
python main.py --input input.mp4 --blur-strength 61 --detection-interval 15 --padding-ratio 0.15 --preview
```

Arguments:

- `--input`: input video file path, required
- `--output`: output video file path, optional; defaults to `<input>_blurred.mp4`
- `--blur-strength`: Gaussian blur kernel size, odd integer, default `51`
- `--detection-interval`: number of frames between face re-detection passes, default `15`
- `--padding-ratio`: extra padding around each detected face before blur is applied, default `0.15`
- `--preview`: show a live preview window; press `Q` to cancel

## How It Works

1. The script validates the input video and opens it with OpenCV.
2. The first frame is run through the face detector.
3. A tracker is initialized for each detected face.
4. On later frames, the tracker updates the face locations until the next detection pass.
5. Each face box is blurred and written to a temporary output video.
6. When the video ends, the output is saved and audio is muxed back in if `ffmpeg` is installed.

## Example

```bash
python main.py --input sample.mp4 --output sample_blurred.mp4 --blur-strength 51 --detection-interval 15
```

## Notes

- The `yolov8n-face.pt` model is resolved automatically when possible.
- If the model file is unavailable, the detector falls back to OpenCV Haar cascades.
- If `ffmpeg` is missing, the script still saves the processed video without original audio.
