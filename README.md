# Blurry Slurry

Blurry Slurry is an app that detects faces in videos and images, tracks them across frames in videos, and applies a configurable blur before exporting the finished file.

It ships as a downloadable `.exe` for Windows and includes an easy drag-and-drop interface for selecting a file, adjusting blur settings, and processing the media.

## Features

- YOLOv8 face detection (`yolov8n-face.pt`) via `ultralytics`
- CSRT tracker updates between detection passes for performance
- Re-detection every configurable N frames to reduce tracker drift
- Gaussian blur applied per tracked face ROI
- Optional real-time preview (`Q` to cancel)
- Windows desktop app experience with a bundled `.exe`
- Drag-and-drop image and video upload interface
- Progress bar per frame using `tqdm`
- Audio preservation by muxing original audio with `ffmpeg`
- Graceful fallback when `ffmpeg` is missing (video still saved)

## Project Structure

- `main.py`: core processing pipeline and CLI entry point
- `app.py`: Streamlit front end for interactive image and video processing
- `run_app.py`: bundled Streamlit launcher used by the executable
- `build_exe.spec`: PyInstaller build definition
- `build_exe.bat`: Windows build script
- `detector.py`: YOLOv8 face detector wrapper
- `tracker.py`: CSRT tracker management
- `blurrer.py`: Face blur logic
- `preview.py`: OpenCV preview window helpers
- `utils.py`: Validation, progress, and audio mux helpers

## Download

Download the latest release for Windows and run `BlurrySlurry.exe`.

Install `ffmpeg` if you want the app to preserve the original audio track in exported videos.

## How To Use

1. Open `BlurrySlurry.exe`.
2. Drag and drop an image or video, or choose a file.
3. Adjust blur strength, radius, and detection interval.
4. Start processing and download the finished file.

## Windows App

Open `BlurrySlurry.exe` on Windows, then drag and drop an image or video or use the file picker, choose your blur strength, radius, and detection interval, and process the media.

If you prefer to run it from source while developing, use:

```bash
python run_app.py
```

## Build From Source

Use this section only if you want to develop or package the app yourself.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then run the app locally with:

```bash
python run_app.py
```

## Build the Windows App

To package the app into a Windows executable, install the requirements and run:

```bash
build_exe.bat
```

The packaged file will be created in the `dist` folder as `BlurrySlurry.exe`.

The first build includes the full Python runtime and project dependencies, so the executable can be copied to another Windows system and run without installing Python separately. It still needs access to the bundled model file and, for audio preservation, `ffmpeg` on the target machine.

### App Controls

- Blur strength: controls how aggressive the face blur looks
- Radius: adds extra padding around each face before blur is applied
- Detection interval: how often the app re-runs face detection

## Example

```bash
python main.py --input input.mp4 --output output.mp4 --blur-strength 61 --preview --detection-interval 15
```

## Notes

- The YOLO weights are loaded from `yolov8n-face.pt`; the model will download automatically on first run when needed.
- If no faces are detected in a frame, processing continues and that frame is written unchanged.

