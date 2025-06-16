#!/usr/bin/env python3
"""
Test script to verify video orientation handling
"""
import cv2
import argparse
from pathlib import Path

def test_video_orientation(video_path):
    """Test video orientation properties"""
    print(f"Testing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    
    print(f"Original video properties:")
    print(f"  Dimensions: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frame count: {frame_count}")
    print(f"  Orientation metadata: {orientation}°")
    print(f"  Is portrait: {height > width}")
    
    # Read first frame
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"  First frame actual dimensions: {w}x{h}")
        print(f"  Frame is portrait: {h > w}")
        
        # Test rotation logic
        def rotate_frame(im, orientation):
            if orientation == 90:
                return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 180:
                return cv2.rotate(im, cv2.ROTATE_180)
            elif orientation == 270:
                return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return im
        
        rotated_frame = rotate_frame(frame, orientation)
        rh, rw = rotated_frame.shape[:2]
        print(f"  After rotation: {rw}x{rh}")
        print(f"  Rotated frame is portrait: {rh > rw}")
    
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test video orientation")
    parser.add_argument("video", type=str, help="Path to video file")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found")
        exit(1)
    
    test_video_orientation(video_path)
