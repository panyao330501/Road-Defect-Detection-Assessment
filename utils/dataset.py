import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class RoadDefectDataset(Dataset):
    def __init__(self, root_dir, transforms=None, is_train=True):
        self.root = root_dir
        self.transforms = transforms

        # Assuming structure: data/images/train and data/labels/train
        mode = 'train' if is_train else 'val'
        self.img_dir = os.path.join(root_dir, 'images', mode)
        self.label_dir = os.path.join(root_dir, 'labels', mode)

        # Load all image files
        self.imgs = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Class mapping
        self.classes = ['background', 'defect', 'patch']  # index 0 is reserved for background in torchvision

    def __getitem__(self, idx):
        # Load image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        # Load label (YOLO format: class x_center y_center w h)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])

                # Convert YOLO format (normalized) to Pascal VOC (xyxy)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h

                boxes.append([x1, y1, x2, y2])
                # Shift class ID by 1 because 0 is background in Faster-RCNN/SSD
                labels.append(cls_id + 1)

                # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # TODO: Add more aggressive augmentation here later
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)


# Simple collate function for dataloader (needed for detection tasks)
def collate_fn(batch):
    return tuple(zip(*batch))