# Face Blur (YOLOv8 + OpenCV)

A production-ready Python CLI that detects and blurs faces in a video using YOLOv8 face detection and OpenCV tracking.

This project also includes a Streamlit front end for drag-and-drop uploads and interactive controls.

## Features

- YOLOv8 face detection (`yolov8n-face.pt`) via `ultralytics`
- CSRT tracker updates between detection passes for performance
- Re-detection every configurable N frames to reduce tracker drift
- Gaussian blur applied per tracked face ROI
- Optional real-time preview (`Q` to cancel)
- Streamlit web UI for drag-and-drop video uploads
- Progress bar per frame using `tqdm`
- Audio preservation by muxing original audio with `ffmpeg`
- Graceful fallback when `ffmpeg` is missing (video still saved)

## Project Structure

- `main.py`: CLI entry point and pipeline orchestration
- `app.py`: Streamlit front end for interactive video processing
- `detector.py`: YOLOv8 face detector wrapper
- `tracker.py`: CSRT tracker management
- `blurrer.py`: Face blur logic
- `preview.py`: OpenCV preview window helpers
- `utils.py`: Validation, progress, and audio mux helpers

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Install `ffmpeg` and ensure it is available in your `PATH` if you want to preserve original audio.

## Usage

```bash
python main.py --input <path> --output <path> [--blur-strength <int>] [--preview] [--detection-interval <int>]
```

## Web UI

```bash
streamlit run app.py
```

The UI lets you drag and drop a video, adjust blur strength, adjust the face padding radius, and change the detection interval before exporting the result.

If you run `python app.py` directly, it will now hand off to Streamlit automatically.

### Arguments

- `--input`: required path to input video
- `--output`: output path (default: `<input>_blurred.mp4`)
- `--blur-strength`: Gaussian kernel size, odd integer (default: `51`)
- `--preview`: show live preview window; press `Q` to cancel processing
- `--detection-interval`: frames between YOLO re-detection (default: `15`)
- `--padding-ratio`: extra padding around detected faces before blur is applied (default: `0.15`)

## Example

```bash
python main.py --input input.mp4 --output output.mp4 --blur-strength 61 --preview --detection-interval 15
```

## Notes

- The YOLO weights are loaded from `yolov8n-face.pt`; the model will download automatically on first run when needed.
- If no faces are detected in a frame, processing continues and that frame is written unchanged.
