from __future__ import annotations

import tempfile
import subprocess
import sys
from pathlib import Path

from main import process_video_file


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
    page_title="Face Blur Studio",
    page_icon="🎯",
    layout="wide",
)


st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 196, 112, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(91, 141, 236, 0.14), transparent 24%),
            linear-gradient(180deg, #0f1115 0%, #151922 100%);
        color: #f4f7fb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 1.2rem;
        background: rgba(12, 15, 22, 0.72);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        color: #fff8ef;
        letter-spacing: -0.04em;
    }
    .hero-subtitle {
        margin-top: 0.5rem;
        color: rgba(244, 247, 251, 0.76);
        line-height: 1.5;
    }
    .feature-pill {
        display: inline-block;
        margin: 0.4rem 0.5rem 0 0;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: #f4f7fb;
        font-size: 0.84rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero-card">
  <h1 class="hero-title">Face Blur Studio</h1>
  <div class="hero-subtitle">
    Upload a video or drag and drop one here, then tune blur strength, radius, and detection interval before exporting the blurred result.
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
        "Select a video file",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        accept_multiple_files=False,
    )
    blur_strength = st.slider("Blur strength", min_value=3, max_value=121, value=51, step=2)
    radius = st.slider("Radius", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    detection_interval = st.slider("Detection interval", min_value=1, max_value=60, value=15, step=1)
    output_name = st.text_input("Output filename", value="blurred_output.mp4")

with right:
    st.subheader("Preview")
    st.caption("The uploader supports drag and drop. After processing, the video appears here with a download button.")

if "result_notice" not in st.session_state:
    st.session_state.result_notice = ""
if "result_bytes" not in st.session_state:
    st.session_state.result_bytes = None
if "result_name" not in st.session_state:
    st.session_state.result_name = "blurred_output.mp4"


def run_processing() -> None:
    if uploaded_file is None:
        st.error("Choose a video file first.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / uploaded_file.name
        output_path = temp_dir_path / output_name

        input_path.write_bytes(uploaded_file.getbuffer())

        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(current: int, total: int) -> None:
            if total > 0:
                progress_bar.progress(min(current / total, 1.0))
            status.write(f"Processing frame {current} of {total}...")

        try:
            result = process_video_file(
                input_path,
                output_path,
                blur_strength=blur_strength,
                detection_interval=detection_interval,
                padding_ratio=radius,
                preview=False,
                progress_callback=update_progress,
            )
        except Exception as exc:
            st.session_state.result_bytes = None
            st.error(f"Processing failed: {exc}")
            return

        status.write("Processing complete.")
        progress_bar.progress(1.0)
        st.session_state.result_notice = (
            f"Saved to {result.output_path.name} in {result.elapsed_seconds:.2f}s"
        )
        st.session_state.result_name = result.output_path.name
        st.session_state.result_bytes = result.output_path.read_bytes()

        st.download_button(
            "Download blurred video",
            data=st.session_state.result_bytes,
            file_name=st.session_state.result_name,
            mime="video/mp4",
            use_container_width=True,
        )


if st.button("Process video", type="primary", use_container_width=True):
    run_processing()

if st.session_state.result_bytes:
    st.success(st.session_state.result_notice)
    st.video(st.session_state.result_bytes)
    st.download_button(
        "Download last result",
        data=st.session_state.result_bytes,
        file_name=st.session_state.result_name,
        mime="video/mp4",
        use_container_width=True,
    )