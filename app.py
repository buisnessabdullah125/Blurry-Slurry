from __future__ import annotations

import tempfile
import subprocess
import sys
from pathlib import Path

from main import process_image_file, process_video_file
from utils import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS


def _ensure_streamlit_runtime() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = None

    if get_script_run_ctx is None or get_script_run_ctx() is not None:
        return

    script_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)], check=False)
    raise SystemExit(0)


_ensure_streamlit_runtime()

import streamlit as st


st.set_page_config(
    page_title="Blurry Slurry",
    page_icon="🎯",
    layout="wide",
)


st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(74, 222, 128, 0.16), transparent 26%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.14), transparent 22%),
            linear-gradient(180deg, #08110b 0%, #0d1710 100%);
        color: #f4f7fb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(74, 222, 128, 0.20);
        border-radius: 1.2rem;
        background: rgba(8, 17, 11, 0.78);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        color: #eafff1;
        letter-spacing: -0.04em;
    }
    .hero-subtitle {
        margin-top: 0.5rem;
        color: rgba(234, 255, 241, 0.76);
        line-height: 1.5;
    }
    .feature-pill {
        display: inline-block;
        margin: 0.4rem 0.5rem 0 0;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(74, 222, 128, 0.14);
        border: 1px solid rgba(74, 222, 128, 0.25);
        color: #d8ffe6;
        font-size: 0.84rem;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #2fbf71 0%, #16a34a 100%);
        color: white;
        border: none;
        border-radius: 0.9rem;
        box-shadow: 0 10px 28px rgba(22, 163, 74, 0.35);
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #35cc78 0%, #12803b 100%);
        color: white;
        border: none;
    }
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero-card">
    <h1 class="hero-title">Blurry Slurry</h1>
  <div class="hero-subtitle">
        Upload a video or image, then tune blur strength and radius before exporting the blurred result.
  </div>
  <span class="feature-pill">Drag & drop upload</span>
  <span class="feature-pill">YOLO face detection</span>
  <span class="feature-pill">OpenCV tracking</span>
</div>
""",
    unsafe_allow_html=True,
)


left, right = st.columns([1.05, 0.95], gap="large")

with left:
    uploaded_file = st.file_uploader(
        "Select an image or video file",
        type=sorted(SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS),
        accept_multiple_files=False,
    )
    blur_strength = st.slider("Blur strength", min_value=3, max_value=121, value=51, step=2)
    radius = st.slider("Radius", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    detection_interval = st.slider("Detection interval", min_value=1, max_value=60, value=15, step=1)
    media_suffix = Path(uploaded_file.name).suffix.lower() if uploaded_file else ".mp4"
    default_output_name = "blurry_slurry_output" + (media_suffix if media_suffix in SUPPORTED_IMAGE_EXTENSIONS else ".mp4")
    output_name = st.text_input("Output filename", value=default_output_name)

with right:
    st.subheader("Preview")
    st.caption("The uploader supports drag and drop. After processing, the image or video appears here with a download button.")
    if uploaded_file is not None:
        preview_bytes = uploaded_file.getvalue()
        if Path(uploaded_file.name).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            st.image(preview_bytes, caption=uploaded_file.name, use_container_width=True)
        else:
            st.video(preview_bytes)

if "result_notice" not in st.session_state:
    st.session_state.result_notice = ""
if "result_bytes" not in st.session_state:
    st.session_state.result_bytes = None
if "result_name" not in st.session_state:
    st.session_state.result_name = "blurry_slurry_output.mp4"
if "result_mime" not in st.session_state:
    st.session_state.result_mime = "video/mp4"


def run_processing() -> None:
    if uploaded_file is None:
        st.error("Choose an image or video file first.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / uploaded_file.name
        output_path = temp_dir_path / output_name

        input_path.write_bytes(uploaded_file.getbuffer())

        progress_bar = st.progress(0)
        status = st.empty()
        suffix = input_path.suffix.lower()

        def update_progress(current: int, total: int) -> None:
            if total > 0:
                progress_bar.progress(min(current / total, 1.0))
            status.write(f"Processing frame {current} of {total}...")

        try:
            if suffix in SUPPORTED_IMAGE_EXTENSIONS:
                result = process_image_file(
                    input_path,
                    output_path,
                    blur_strength=blur_strength,
                    padding_ratio=radius,
                )
                st.session_state.result_mime = f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix.lstrip('.')}"
            else:
                result = process_video_file(
                    input_path,
                    output_path,
                    blur_strength=blur_strength,
                    detection_interval=detection_interval,
                    padding_ratio=radius,
                    preview=False,
                    progress_callback=update_progress,
                )
                st.session_state.result_mime = "video/mp4"
        except Exception as exc:
            st.session_state.result_bytes = None
            st.error(f"Processing failed: {exc}")
            return

        status.write("Processing complete.")
        progress_bar.progress(1.0)
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            st.session_state.result_notice = (
                f"Saved to {result.output_path.name} in {result.elapsed_seconds:.2f}s"
            )
        else:
            st.session_state.result_notice = (
                f"Saved to {result.output_path.name} in {result.elapsed_seconds:.2f}s"
            )
        st.session_state.result_name = result.output_path.name
        st.session_state.result_bytes = result.output_path.read_bytes()

        st.download_button(
            "Download blurred file",
            data=st.session_state.result_bytes,
            file_name=st.session_state.result_name,
            mime=st.session_state.result_mime,
            use_container_width=True,
        )


if st.button("Process image/video", type="primary", use_container_width=True):
    run_processing()

if st.session_state.result_bytes:
    st.success(st.session_state.result_notice)
    if st.session_state.result_mime.startswith("image/"):
        st.image(st.session_state.result_bytes, use_container_width=True)
    else:
        st.video(st.session_state.result_bytes)
    st.download_button(
        "Download last result",
        data=st.session_state.result_bytes,
        file_name=st.session_state.result_name,
        mime=st.session_state.result_mime,
        use_container_width=True,
    )