import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDHead


def get_baseline_model(model_name='faster_rcnn', num_classes=3):

    if model_name == 'faster_rcnn':
        # Load pre-trained model on COCO
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

        # Replace the classifier head with a new one for our custom classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    elif model_name == 'ssd':
        # Load SSD300 with VGG16 backbone
        model = ssd300_vgg16(weights='DEFAULT')

        # Re-initialize the head
        in_channels = [512, 1024, 512, 256, 256, 256]  # VGG16 feature map sizes
        num_anchors = model.anchor_generator.num_anchors_per_location()

        # Reset head for custom classes
        model.head = SSDHead(in_channels, num_anchors, num_classes)

        return model

    else:
        raise ValueError(f"Model {model_name} not supported")


if __name__ == '__main__':
    # Test code to check input/output shapes
    net = get_baseline_model('ssd', 3)
    print(net)