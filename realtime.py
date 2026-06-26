import cv2
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

MODEL_PATH = Path("face_model.yml")
LABELS_PATH = Path("labels.json")

GREETINGS = {
    "ayan":     "Hi Ayan, how are you doing?",
    "nakib":    "Hello Nakib!",
    "mehjabin": "Hello Mehjabin!",
}
UNKNOWN_MSG = "I'm sorry, I can't recognize you."
CONFIDENCE_THRESHOLD = 80  # LBPH: lower = better; above this = unknown
SPEAK_COOLDOWN = 6.0       # seconds before repeating same greeting

_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_recognizer = cv2.face.LBPHFaceRecognizer_create()
_label_map: dict[int, str] = {}
_last_spoken: dict[str, float] = {}


def _load_model() -> bool:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("Model not found. Train first: POST http://localhost:8000/train")
        return False
    _recognizer.read(str(MODEL_PATH))
    _label_map.update(
        {int(k): v for k, v in json.loads(LABELS_PATH.read_text()).items()}
    )
    return True


def _speak(text: str):
    subprocess.Popen(["say", text])


def _maybe_speak(key: str, message: str):
    now = time.time()
    if now - _last_spoken.get(key, 0) > SPEAK_COOLDOWN:
        _last_spoken[key] = now
        threading.Thread(target=_speak, args=(message,), daemon=True).start()


def run():
    if not _load_model():
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("Real-time face recognition running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            label, confidence = _recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                raw_name = _label_map.get(label, "unknown").lower()
                greeting = GREETINGS.get(raw_name, f"Hello {raw_name.title()}!")
                display = raw_name.title()
                color = (0, 200, 0)
                speak_key = raw_name
            else:
                greeting = UNKNOWN_MSG
                display = "Unknown"
                color = (0, 0, 220)
                speak_key = "unknown"

            _maybe_speak(speak_key, greeting)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, display, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(frame, greeting, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
