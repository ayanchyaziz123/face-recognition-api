import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH      = Path("face_dl_model.pth")
ATTENDANCE_CSV  = Path("attendance.csv")
UNKNOWN_DIR     = Path("unknown")
CONFIDENCE      = 0.4
LOG_INTERVAL    = 60.0   # seconds before logging same person again
UNK_INTERVAL    = 10.0   # seconds before saving another unknown snapshot

UNKNOWN_DIR.mkdir(exist_ok=True)

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


def _log_attendance(name: str, confidence: float):
    exists = ATTENDANCE_CSV.exists()
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["name", "timestamp", "confidence"])
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(confidence, 3)])


def _save_unknown(face_bgr):
    filename = UNKNOWN_DIR / f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(filename), face_bgr)


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
        self._log_cooldown:  dict[str, float] = {}
        self._unk_last_saved: float = 0
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
        now = time.time()

        for x, y, w, h in faces:
            face_bgr = bgr[y:y+h, x:x+w]
            rgb      = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            tensor   = transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1)[0]

            conf, idx = probs.max(0)
            conf_val  = conf.item()

            if conf_val >= CONFIDENCE:
                name     = self.classes[idx.item()]
                greeting = GREETINGS.get(name.lower(), f"Hello {name}!")

                # Log attendance with cooldown
                last = self._log_cooldown.get(name.lower(), 0)
                if now - last >= LOG_INTERVAL:
                    _log_attendance(name, conf_val)
                    self._log_cooldown[name.lower()] = now
            else:
                name     = "Unknown"
                greeting = "I'm sorry, I can't recognize you."

                # Save unknown face snapshot with cooldown
                if now - self._unk_last_saved >= UNK_INTERVAL:
                    _save_unknown(face_bgr)
                    self._unk_last_saved = now

            results.append({
                "name":       name,
                "greeting":   greeting,
                "confidence": round(conf_val, 3),
                "bbox":       {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            })

        return {"face_detected": bool(results), "results": results}
