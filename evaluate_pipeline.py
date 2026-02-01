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
        device='cuda'
):
    # 1. Load Models
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load YOLOv5
    print(f"Loading YOLOv5 from {weights_yolo}...")
    model_yolo = DetectMultiBackend(weights_yolo, device=device)
    stride, names, pt = model_yolo.stride, model_yolo.names, model_yolo.pt

    # Load U-Net
    print(f"Loading U-Net from {weights_unet}...")
    model_unet = UNet(n_channels=3, n_classes=1).to(device)
    # model_unet.load_state_dict(torch.load(weights_unet, map_location=device))
    model_unet.eval()

    # 2. Process Video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error reading video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess for YOLO
        img = letterbox(frame, 640, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]

        # YOLO Inference
        pred = model_yolo(img)
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None)

        # Process detections
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes to original image
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)]

                    # Only process if it is a 'defect' (class 0), ignore 'patch' (class 1)
                    if int(cls) == 0:
                        x1, y1, x2, y2 = map(int, xyxy)

                        # Crop the defect area
                        defect_crop = frame[y1:y2, x1:x2]

                        if defect_crop.size == 0: continue

                        # Step 3: IPM Transform (Perspective Correction)
                        # TODO: Optimize IPM to only transform the crop context
                        ipm_img, _ = get_ipm_transform(defect_crop)

                        # Step 4: U-Net Segmentation
                        # Preprocess for U-Net
                        u_input = cv2.resize(ipm_img, (256, 256))  # Resize to UNet input
                        u_input = torch.from_numpy(u_input.transpose(2, 0, 1)).float().to(device) / 255.0
                        u_input = u_input.unsqueeze(0)

                        with torch.no_grad():
                            u_pred = model_unet(u_input)
                            u_pred = torch.sigmoid(u_pred)
                            u_mask = (u_pred > 0.5).float().cpu().numpy()[0][0] * 255

                        # Step 5: Area Calculation
                        area = calculate_defect_area(u_mask)

                        # Draw bounding box and severity
                        color = (0, 0, 255)  # Red for defect
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} Area:{area:.1f}cm2", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    else:
                        # Draw Patch (Green) - visualized but not evaluated
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, "Patch (Ignored)", (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show result
        # cv2.imshow('Detection & Evaluation', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images/sample.mp4', help='source')
    parser.add_argument('--weights-yolo', type=str, default='weights/best_defect.pt', help='yolo weights path')
    opt = parser.parse_args()

    # run_pipeline(**vars(opt))
    print("Pipeline loaded. Connect weights to run.")