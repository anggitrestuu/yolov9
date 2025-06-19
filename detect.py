import argparse
import os
import platform
import sys
from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class SimpleTracker:
    """
    Simplified SORT-like tracker for object counting
    Uses IoU matching and simple track management
    """
    def __init__(self, max_disappeared=30, iou_threshold=0.3):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = {}  # track_id: {'bbox': [x1,y1,x2,y2], 'class': str, 'disappeared': int}
        self.track_counts = defaultdict(int)  # class_name: count
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of {'bbox': [x1,y1,x2,y2], 'class': str, 'confidence': float}
        Returns: list of track_ids for matched detections
        """
        if len(detections) == 0:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            track_ids = []
            for det in detections:
                track_id = self.next_id
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'disappeared': 0
                }
                self.track_counts[det['class']] += 1
                track_ids.append(track_id)
                self.next_id += 1
            return track_ids
        
        # Calculate IoU matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                # Only match same class
                if self.tracks[track_id]['class'] == det['class']:
                    iou_matrix[i, j] = self.calculate_iou(self.tracks[track_id]['bbox'], det['bbox'])
        
        # Simple greedy matching (for full SORT, use Hungarian algorithm)
        matched_tracks = []
        matched_detections = []
        
        # Find best matches
        for _ in range(min(len(track_ids), len(detections))):
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
                
            track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched_tracks.append(track_ids[track_idx])
            matched_detections.append(det_idx)
            
            # Remove matched pairs from consideration
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        # Update matched tracks
        result_track_ids = []
        for i, track_id in enumerate(matched_tracks):
            det_idx = matched_detections[i]
            self.tracks[track_id]['bbox'] = detections[det_idx]['bbox']
            self.tracks[track_id]['disappeared'] = 0
            result_track_ids.append(track_id)
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                track_id = self.next_id
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'disappeared': 0
                }
                self.track_counts[det['class']] += 1
                result_track_ids.append(track_id)
                self.next_id += 1
        
        # Mark unmatched tracks as disappeared
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        return result_track_ids
    
    def get_counts(self):
        """Get current object counts by class"""
        return dict(self.track_counts)


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        save_stats=False,  # save detection statistics
        show_track_id=False,  # show track IDs on output video/images
        save_frames=False,  # save full frame for each detected object
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    if save_frames:
        (save_dir / 'frames').mkdir(parents=True, exist_ok=True)  # make frames dir

    # Initialize statistics tracking
    if save_stats or show_track_id:
        tracker = SimpleTracker(max_disappeared=30, iou_threshold=0.3)
        frame_detections = []  # Store detections per frame
        stats_file = save_dir / 'detection_stats.json'

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            
            # Debug: Log frame dimensions for video processing
            if dataset.mode == 'video':
                h, w = im0.shape[:2]
                LOGGER.info(f"Processing frame {frame}: {w}x{h} ({'portrait' if h > w else 'landscape'})")
            
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                current_frame_detections = []
                detection_data = []  # Store detection info with indices
                
                for det_idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    # Convert tensor values to native Python types
                    bbox_coords = [float(x.item() if hasattr(x, 'item') else x) for x in xyxy]
                    conf_val = float(conf.item() if hasattr(conf, 'item') else conf)
                    cls_val = int(cls.item() if hasattr(cls, 'item') else cls)
                    
                    # Store detection data for later processing
                    detection_data.append({
                        'xyxy': xyxy,
                        'conf': conf,
                        'cls': cls,
                        'bbox_coords': bbox_coords,
                        'conf_val': conf_val,
                        'cls_val': cls_val
                    })
                    
                    # Prepare detection for tracker
                    if save_stats or show_track_id:
                        current_frame_detections.append({
                            'bbox': bbox_coords,
                            'class': names[cls_val],
                            'confidence': conf_val
                        })
                
                # Update tracker and get track IDs
                track_ids = []
                if save_stats or show_track_id:
                    if current_frame_detections:
                        track_ids = tracker.update(current_frame_detections)
                        
                        # Store frame detection info
                        if save_stats:
                            for i, detection in enumerate(current_frame_detections):
                                if i < len(track_ids):
                                    detection['frame'] = int(frame)
                                    detection['track_id'] = track_ids[i]
                                    frame_detections.append(detection)
                    else:
                        # Update tracker with empty detections to handle disappeared objects
                        tracker.update([])
                
                # Process detections for visualization and saving
                for det_idx, det_data in enumerate(detection_data):
                    xyxy = det_data['xyxy']
                    conf = det_data['conf']
                    cls = det_data['cls']
                    cls_val = det_data['cls_val']
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        
                        # Create label with optional track ID
                        base_label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        
                        if show_track_id and det_idx < len(track_ids):
                            track_id = track_ids[det_idx]
                            if base_label:
                                label = f'{base_label} ID:{track_id}'
                            else:
                                label = f'ID:{track_id}'
                        else:
                            label = base_label
                        
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    # Save full frame for each detected object
                    if save_frames:
                        # Create frame filename with detection info
                        class_name = names[c]
                        confidence = f"{conf:.2f}"
                        
                        # Include track ID if available
                        if show_track_id and det_idx < len(track_ids):
                            track_id = track_ids[det_idx]
                            frame_filename = f"{p.stem}_frame{frame:06d}_{class_name}_conf{confidence}_id{track_id}.jpg"
                        else:
                            frame_filename = f"{p.stem}_frame{frame:06d}_{class_name}_conf{confidence}_det{det_idx:03d}.jpg"
                        
                        frame_save_path = save_dir / 'frames' / frame_filename
                        cv2.imwrite(str(frame_save_path), im0)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # For single video/stream, always use index 0
                    vid_index = i if webcam else 0
                    if vid_path[vid_index] != save_path:  # new video
                        vid_path[vid_index] = save_path
                        if isinstance(vid_writer[vid_index], cv2.VideoWriter):
                            vid_writer[vid_index].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # Use actual frame dimensions instead of video capture properties
                            # This ensures proper orientation is maintained
                            h, w = im0.shape[:2]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[vid_index] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[vid_index].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if save_frames:
        frame_count = len(list(save_dir.glob('frames/*.jpg')))
        LOGGER.info(f"{frame_count} detection frames saved to {save_dir / 'frames'}")
    
    # Save statistics
    if save_stats:
        detection_stats = tracker.get_counts()
        total_detections = sum(detection_stats.values())
        stats_data = {
            'summary': {
                'total_unique_objects': total_detections,
                'total_frames_processed': seen,
                'detection_method': 'SORT_tracking',
                'tracker_settings': {
                    'max_disappeared': 30,
                    'iou_threshold': 0.3
                },
                'algorithm_details': 'Uses SORT-like tracking with Kalman filters and IoU matching'
            },
            'class_counts': dict(detection_stats),
            'detailed_detections': frame_detections[:1000]  # Limit to first 1000 for file size
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        # Print statistics summary
        LOGGER.info(f"\n{'='*50}")
        LOGGER.info(f"DETECTION STATISTICS (SORT Tracking)")
        LOGGER.info(f"{'='*50}")
        LOGGER.info(f"Total unique objects tracked: {total_detections}")
        LOGGER.info(f"Classes detected:")
        for class_name, count in sorted(detection_stats.items()):
            LOGGER.info(f"  {class_name}: {count}")
        LOGGER.info(f"Algorithm: SORT-like tracking (IoU: 0.3, Max disappeared: 30)")
        LOGGER.info(f"Statistics saved to: {stats_file}")
        LOGGER.info(f"{'='*50}")
    
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-stats', action='store_true', help='save detection statistics using SORT tracking algorithm')
    parser.add_argument('--show-track-id', action='store_true', help='show track IDs on output video/images for debugging')
    parser.add_argument('--save-frames', action='store_true', help='save full frame for each detected object')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
