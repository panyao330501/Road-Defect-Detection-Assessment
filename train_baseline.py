"""
baseline train
"""
import torch
from utils.road_dataset import RoadDefectDataset, collate_fn
from models.baselines import get_baseline_model
from torch.utils.data import DataLoader


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    loss_sum = 0
    for i, (images, targets) in enumerate(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_sum += losses.item()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {losses.item():.4f}")


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on {device}")

    # Dataset
    dataset_train = RoadDefectDataset('data', is_train=True)
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)

    # Setup Model (Change to 'ssd' or 'faster_rcnn')
    model_name = 'faster_rcnn'
    print(f"Initializing {model_name}...")
    model = get_baseline_model(model_name, num_classes=3)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    #training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, loader_train, device, epoch)

        # 10bu 1cun
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"weights/{model_name}_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()