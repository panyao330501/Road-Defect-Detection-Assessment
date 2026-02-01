import argparse
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from models.unet_custom import UNet

# Hardcoded parameters for IPM (Based on camera setup: h=1.5m, angle=25deg)
# src_points and dst_points needs calibration
SRC_POINTS = np.float32([[500, 400], [1420, 400], [0, 1080], [1920, 1080]])
DST_POINTS = np.float32([[0, 0], [400, 0], [100, 600], [300, 600]])


def get_ipm_transform(img):
    """
    Apply Inverse Perspective Mapping to transform road surface to bird's eye view.
    """
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
    warped = cv2.warpPerspective(img, M, (400, 600))  # Fixed size for UNet input
    return warped, M


def calculate_defect_area(mask, pixel_ratio=0.05):
    """
    Calculate physical area from binary mask.
    pixel_ratio: cm^2 per pixel (calibrated value)
    """
    white_pixels = np.sum(mask == 255)
    area = white_pixels * pixel_ratio
    return area


def run_pipeline(
        weights_yolo='weights/best_defect.pt',
        weights_unet='weights/unet_road.pth',
        source='data/images/test_video.mp4',
        conf_thres=0.4,
        device='cuda',
        save_dir='runs/evaluation'  # [New] Add save directory
):
    # Create output directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_file = save_path / 'result_severity.mp4'  # Output video path

    # ... (Load Models code is same) ...

    # 2. Process Video setup
    cap = cv2.VideoCapture(source)
    # Get video properties for writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video Writer
    out_writer = cv2.VideoWriter(str(save_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ... (YOLO inference and Processing code is same) ...

        # [Inside the loop, after drawing boxes]
        # Instead of cv2.imshow, write to file
        out_writer.write(frame)

    cap.release()
    out_writer.release()  # Save video
    print(f"Processing complete. Results saved to {save_file}")


if __name__ == "__main__":
    # ... (args setup) ...
    # Default behavior saves to runs/evaluation
    print("Pipeline initialized...")