import os
import logging
from pathlib import Path
from typing import NamedTuple
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
from io import BytesIO

# Add YOLOv9 root to path
HERE = Path(__file__).parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

MODEL_PATH = ROOT / "./rdd/best.pt"

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

st.title("Road Damage Object Detection - Image")
st.write("Detect Road Damage objects in an image. Upload the image and start detecting.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Adjust the confidence threshold to control detection sensitivity.")

if image_file is not None:
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    # Print string for logging
    s = ''
    
    # Convert to numpy array
    img0 = np.array(image)
    h_ori, w_ori = img0.shape[:2]

    # Resize and prepare image
    imgsz = (640, 640)
    stride = max(int(model.stride), 32) if hasattr(model, 'stride') else 32
    imgsz = check_img_size(imgsz, s=stride)
    
    # Print string for logging
    s = ''
    s += f'{imgsz[0]}x{imgsz[1]} '  # menggunakan f-string dengan akses individual ke elemen imgsz
    
    # Initialize timing array
    dt = (Profile(), Profile(), Profile())
    
    # Preprocess timing
    with dt[0]:
        img = cv2.resize(img0, imgsz, interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

    # Inference timing
    with dt[1]:
        with torch.no_grad():
            pred = model(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0][1]  # YOLOv9 returns (pred_distill, pred) where pred_distill is a tuple
    
    # NMS timing
    with dt[2]:
        pred = non_max_suppression(pred, score_threshold, 0.45, max_det=1000)

    # Process detections
    detections = []
    # Initialize annotator outside the if condition
    annotator = Annotator(img0.copy(), line_width=3)
    
    if len(pred[0]):
        # Rescale boxes from img_size to im0 size
        pred[0][:, :4] = scale_boxes(img.shape[2:], pred[0][:, :4], img0.shape).round()
        
        # Create detection summary
        detection_summary = {}
        for *xyxy, conf, cls in reversed(pred[0]):
            c = int(cls)
            label = f'{CLASSES[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
            
            # Count detections per class
            if CLASSES[c] not in detection_summary:
                detection_summary[CLASSES[c]] = 0
            detection_summary[CLASSES[c]] += 1
            
            detections.append(
                Detection(
                    class_id=c,
                    label=CLASSES[c],
                    score=float(conf),
                    box=torch.tensor(xyxy).numpy()
                )
            )
        
        # Print results per class
        for c in pred[0][:, 5].unique():
            n = (pred[0][:, 5] == c).sum()  # detections per class
            s += f"{n} {CLASSES[int(c)]}{'s' * (n > 1)}, "  # add to string
        
        # Display detection summary
        st.sidebar.markdown("### Detection Summary")
        st.sidebar.write(s.rstrip(", "))
        
        # Display performance metrics
        t = tuple(x.t / 1 * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        
        st.sidebar.markdown("### Performance")
        st.sidebar.write(f"Image size: {imgsz[0]}x{imgsz[1]}")
        st.sidebar.write(f"Pre-process: {t[0]:.1f}ms")
        st.sidebar.write(f"Inference: {t[1]:.1f}ms")
        st.sidebar.write(f"NMS: {t[2]:.1f}ms")
        total_time = sum(t)
        st.sidebar.write(f"Total: {total_time:.1f}ms")
        st.sidebar.write(f"Speed: {1000/total_time:.1f} FPS")

    else:
        LOGGER.info(f"{s}(no detections)")
        st.sidebar.markdown("### Detection Summary")
        st.sidebar.write("No detections found")
        
        # Still show performance metrics
        t = tuple(x.t / 1 * 1E3 for x in dt)
        st.sidebar.markdown("### Performance")
        st.sidebar.write(f"Image size: {imgsz[0]}x{imgsz[1]}")
        st.sidebar.write(f"Pre-process: {t[0]:.1f}ms")
        st.sidebar.write(f"Inference: {t[1]:.1f}ms")
        st.sidebar.write(f"NMS: {t[2]:.1f}ms")
        total_time = sum(t)
        st.sidebar.write(f"Total: {total_time:.1f}ms")
        st.sidebar.write(f"Speed: {1000/total_time:.1f} FPS")

    # Show results
    with col1:
        st.write("#### Original Image")
        st.image(img0)
    
    with col2:
        st.write("#### Predictions")
        result_img = annotator.result()
        st.image(result_img)

        # Download predicted image
        buffer = BytesIO()
        result_pil = Image.fromarray(result_img)
        result_pil.save(buffer, format="PNG")
        result_bytes = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=result_bytes,
            file_name="Road Damage_Detection.png",
            mime="image/png"
        )