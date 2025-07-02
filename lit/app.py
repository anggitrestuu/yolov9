import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
from pathlib import Path
import sys
from PIL import Image
import zipfile
import io

# Add YOLOv9 to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors


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
    """Draw detection boxes on image"""
    annotator = Annotator(image)
    for r in results:
        label = f"{r['name']} {r['conf']:.2f}"
        annotator.box_label(r["xyxy"], label, color=colors(r["cls"], True))
    return annotator.result()


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


def process_batch_images(model, images, confidence, iou):
    """Process multiple images and return results"""
    processed_results = []
    for img in images:
        # Convert PIL Image to cv2 format
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Process image
        results = process_image(model, img_cv2, confidence, iou)

        # Draw results
        output_image = draw_boxes(img_cv2.copy(), results)

        processed_results.append({"image": output_image, "detections": results})
    return processed_results


def create_zip_of_images(images, results):
    """Create a ZIP file containing processed images"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for idx, (img, result) in enumerate(zip(images, results)):
            # Convert numpy array to PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB))

            # Save image to bytes
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            # Add to ZIP
            zip_file.writestr(f"processed_image_{idx+1}.png", img_byte_arr)

            # Create detection results text file
            det_text = f"Detections for image_{idx+1}:\n"
            for det in result["detections"]:
                det_text += f"- {det['name']}: {det['conf']:.2f}\n"
            zip_file.writestr(f"detections_{idx+1}.txt", det_text)

    return zip_buffer


def main():
    st.title("YOLOv9 Object Detection")

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
    weights_path = st.sidebar.text_input("Model Weights Path", "./weights/best.pt")

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
    source_type = st.radio(
        "Select Source", ["Single Image", "Batch Images", "Video", "Webcam"]
    )

    if source_type == "Single Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
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

    elif source_type == "Batch Images":
        uploaded_files = st.file_uploader(
            "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )

        if uploaded_files:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process images
            images = []
            for idx, file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {idx + 1} of {len(uploaded_files)}")

                # Open image
                img = Image.open(file)
                images.append(img)

            # Process all images
            results = process_batch_images(model, images, confidence, iou)

            # Create download button for ZIP file
            zip_buffer = create_zip_of_images(images, results)
            st.download_button(
                label="Download Processed Images",
                data=zip_buffer.getvalue(),
                file_name="processed_images.zip",
                mime="application/zip",
            )

            # Display results in a grid
            cols = st.columns(3)
            for idx, result in enumerate(results):
                with cols[idx % 3]:
                    st.image(
                        cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB),
                        caption=f"Image {idx + 1}",
                    )
                    st.write("Detections:")
                    for det in result["detections"]:
                        st.write(f"- {det['name']}: {det['conf']:.2f}")

            # Clear progress bar and status
            progress_bar.empty()
            status_text.empty()

    elif source_type == "Video":
        uploaded_file = st.file_uploader(
            "Choose a video...", type=["mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = process_image(model, frame, confidence, iou)
                output_frame = draw_boxes(frame.copy(), results)
                stframe.image(output_frame, channels="BGR")

            cap.release()

    elif source_type == "Webcam":
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        stop_button = st.button("Stop")

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            results = process_image(model, frame, confidence, iou)
            output_frame = draw_boxes(frame.copy(), results)
            stframe.image(output_frame, channels="BGR")

        cap.release()


if __name__ == "__main__":
    main()
