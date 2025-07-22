import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Add YOLOv9 to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

# Set page config
st.set_page_config(
    page_title="Road Damage Detection - YOLOv9", page_icon="🛣️", layout="wide"
)


def load_yolov9_model(weights_path, device):
    """Load YOLOv9 model"""
    model = DetectMultiBackend(weights_path, device=device)
    return model


def process_image(model, image, conf_thres=0.25, iou_thres=0.45):
    """Process image through YOLOv9"""
    # Resize and prepare image
    img = letterbox(image)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(model.device)
    img = img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model(img)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    results = []
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()

            # Results
            for *xyxy, conf, cls in reversed(det):
                results.append(
                    {
                        "xyxy": xyxy,
                        "conf": float(conf),
                        "cls": int(cls),
                        "name": model.names[int(cls)],
                    }
                )

    return results


def draw_boxes(image, results):
    """Draw detection boxes on image and resize to 640x640"""
    # Draw boxes first
    annotator = Annotator(image)
    for r in results:
        label = f"{r['name']} {r['conf']:.2f}"
        annotator.box_label(r["xyxy"], label, color=colors(r["cls"], True))

    # Get the annotated image
    result_img = annotator.result()

    # Resize to 640x640
    output_img = cv2.resize(result_img, (640, 640), interpolation=cv2.INTER_LINEAR)

    return output_img


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, r, (dw, dh)


def main():
    st.title("Road Damage Detection using YOLOv9")

    # Sidebar
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    iou = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45)

    # Device selection
    device_option = st.sidebar.radio("Select Device", ["CPU", "GPU"])
    device = torch.device(
        "cuda:0" if device_option == "GPU" and torch.cuda.is_available() else "cpu"
    )

    # Model loading
    weights_path = st.sidebar.text_input("Model Weights Path", "./rdd2022/rdd-yolov9-c-converted.pt")

    # Load model
    @st.cache_resource
    def get_model(weights_path, device):
        return load_yolov9_model(weights_path, device)

    try:
        model = get_model(weights_path, device)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process image
        results = process_image(model, image, confidence, iou)

        # Draw results
        output_image = draw_boxes(image.copy(), results)

        # Display results
        st.image(output_image, channels="BGR", caption="Processed Image")

        # Display detections
        st.write("Detections:")
        for r in results:
            st.write(f"- {r['name']}: {r['conf']:.2f}")


if __name__ == "__main__":
    main()
