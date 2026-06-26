import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from model import FaceRecognitionModel, DEVICE

EPOCHS     = 30
BATCH_SIZE = 16
MODELS_DIR = Path("models")

_train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train(org_id: str = "default", status: dict | None = None) -> dict:
    dataset_dir = Path("dataset") / org_id
    model_path  = MODELS_DIR / f"{org_id}_model.pth"
    MODELS_DIR.mkdir(exist_ok=True)

    if not dataset_dir.exists():
        raise ValueError(f"No dataset found for org: {org_id}")

    if status:
        status["state"] = "loading data"

    full_dataset = datasets.ImageFolder(str(dataset_dir), transform=_train_tf)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    if num_classes < 2:
        raise ValueError("Need at least 2 people in the dataset.")

    val_size   = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_set.dataset = copy.deepcopy(full_dataset)
    val_set.dataset.transform = _val_tf

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FaceRecognitionModel(num_classes).to(DEVICE)
    _resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.backbone.load_state_dict(
        torch.nn.Sequential(*list(_resnet.children())[:-1]).state_dict()
    )
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone[-2:].parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.backbone[-2:].parameters(), "lr": 1e-4},
        {"params": model.head.parameters(),           "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_weights = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        if status:
            status.update({"state": "training", "epoch": epoch, "epochs": EPOCHS})

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(images), labels).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                preds    = model(images.to(DEVICE)).argmax(1).cpu()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_weights)
    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
    }, model_path)

    if status:
        status.update({"state": "done", "val_accuracy": round(best_acc, 4)})

    return {"val_accuracy": round(best_acc, 4), "classes": class_names, "samples": len(full_dataset)}
