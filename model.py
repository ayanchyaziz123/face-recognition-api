from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path("face_dl_model.pth")
CONFIDENCE  = 0.4

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

GREETINGS = {
    "ayan":     "Hi Ayan, how are you doing?",
    "nakib":    "Hello Nakib!",
    "mehjabin": "Hello Mehjabin!",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class Recognizer:
    def __init__(self):
        self.model: FaceRecognitionModel | None = None
        self.classes: list[str] = []
        self.load()

    def load(self):
        if not MODEL_PATH.exists():
            return
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        self.model = FaceRecognitionModel(ckpt["num_classes"]).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.classes = ckpt["class_names"]

    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Train in the notebook first.")

        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Cannot decode image.")

        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = _detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        results = []
        for x, y, w, h in faces:
            rgb    = cv2.cvtColor(bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1)[0]

            conf, idx = probs.max(0)
            if conf.item() >= CONFIDENCE:
                name     = self.classes[idx.item()]
                greeting = GREETINGS.get(name.lower(), f"Hello {name}!")
            else:
                name     = "Unknown"
                greeting = "I'm sorry, I can't recognize you."

            results.append({
                "name":       name,
                "greeting":   greeting,
                "confidence": round(conf.item(), 3),
                "bbox":       {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            })

        return {"face_detected": bool(results), "results": results}
