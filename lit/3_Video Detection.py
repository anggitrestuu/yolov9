import os
import logging
from pathlib import Path
from typing import List, NamedTuple
import sys
import time

import cv2
import numpy as np
import streamlit as st
import torch

# Import YOLOv9 dependencies
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from PIL import Image

# Add YOLOv9 root to path
HERE = Path(__file__).parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

MODEL_PATH = ROOT / "./rdd/rdd-yolov9-c.pt"

# Session-specific caching
cache_key = "rdd-yolov9-c"
device = select_device('cpu')  # Define device here globally

if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    # Initialize model with CPU
    model = DetectMultiBackend(MODEL_PATH, device=device, dnn=False)
    model.warmup(imgsz=(1, 3, 640, 640))  # Warmup
    st.session_state[cache_key] = model

CLASSES = [
    "Longitudinal",
    "Transverse",
    "Aligator",
    "Pathole"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Add batch size configuration
BATCH_SIZE = 4  # Bisa disesuaikan dengan memory yang tersedia

# Add frame skip configuration
FRAME_SKIP = 2  # Proses setiap 2 frame

def processVideo(video_file, score_threshold):
    # Write the file into disk
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if (videoCapture.isOpened() == False):
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize image size and stride
        imgsz = (640, 640)
        stride = max(int(model.stride), 32) if hasattr(model, 'stride') else 32
        imgsz = check_img_size(imgsz, s=stride)

        # Resize video if too large
        if _width > resize_width:
            scale = resize_width / _width
            _width = resize_width
            _height = int(_height * scale)

        st.write("Video Duration :", f"{int(_frame_count/_fps/60)}:{int(_frame_count/_fps%60)}")
        st.write("Width, Height and FPS :", _width, _height, _fps)

        inferenceBarText = "Processing video frames, please wait."
        inferenceBar = st.progress(0, text=inferenceBarText)
        imageLocation = st.empty()

        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        _frame_counter = 0
        processing_times = []
        
        # Process frames in smaller batches for CPU
        frames_batch = []
        original_frames = []
        
        while(videoCapture.isOpened()):
            # Collect frames for batch
            while len(frames_batch) < batch_size:
                ret, frame = videoCapture.read()
                if not ret:
                    break
                    
                if _frame_counter % frame_skip == 0:
                    # Preprocess frame
                    img = cv2.resize(frame, imgsz)
                    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.float() / 255.0
                    if len(img.shape) == 3:
                        img = img[None]
                    frames_batch.append(img)
                    original_frames.append(frame)
                
                _frame_counter += 1
                inferenceBar.progress(min(_frame_counter/_frame_count, 1.0), text=inferenceBarText)
            
            if not frames_batch:
                break
                
            # Process batch
            start_time = time.time()
            
            # Stack frames and run inference
            batch = torch.cat(frames_batch)
            with torch.no_grad():
                pred = model(batch)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0][1]
            
            pred = non_max_suppression(pred, score_threshold, 0.45, max_det=1000)
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            # Process predictions and update video
            for i, (det, frame) in enumerate(zip(pred, original_frames)):
                if len(det):
                    # Scale boxes
                    det[:, :4] = scale_boxes(batch[i].shape[1:], det[:, :4], frame.shape).round()
                    
                    # Annotate
                    annotator = Annotator(frame, line_width=3)
                    
                    # Draw boxes
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = f'{CLASSES[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                
                result_img = annotator.result() if len(det) else frame
                
                # Write frame
                cv2writer.write(result_img)
                
                # Update display occasionally
                if _frame_counter % (frame_skip * 2) == 0:
                    imageLocation.image(result_img)
            
            # Clear batch arrays
            frames_batch.clear()
            original_frames.clear()

        # Calculate and display performance metrics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            fps = batch_size / avg_time  # Account for batch processing
            
            st.sidebar.markdown("### Performance Metrics")
            st.sidebar.write(f"Average batch processing time: {avg_time*1000:.1f}ms")
            st.sidebar.write(f"Effective FPS: {fps:.1f}")
            st.sidebar.write(f"Total frames processed: {_frame_counter}")
            st.sidebar.write(f"Batch size: {batch_size}")
            st.sidebar.write(f"Frame skip: {frame_skip}")

        # Release resources
        videoCapture.release()
        cv2writer.release()
        inferenceBar.empty()

        st.success("Video Processing Complete!")

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

st.title("Road Damage Object Detection - Video")
st.write("Detect Road Damage objects in video. Upload the video and start detecting.")

video_file = st.file_uploader("Upload Video", type=".mp4")
st.caption("There is 1GB limit for video size with .mp4 extension. Resize or cut your video if its bigger than 1GB.")

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Adjust the confidence threshold to control detection sensitivity.")

st.sidebar.markdown("### Processing Options")
batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=8, value=4)
frame_skip = st.sidebar.slider("Process every N frames", min_value=1, max_value=10, value=2)
resize_width = st.sidebar.slider("Target Width", min_value=320, max_value=1280, value=640, step=32)

if video_file is not None:
    if st.button('Process Video', use_container_width=True, type="secondary"):
        _warning = "Processing Video " + video_file.name
        st.warning(_warning)
        processVideo(video_file, score_threshold)