# FaceAttend — Smart Attendance System

A production-ready, multi-tenant face recognition attendance system built with FastAPI, PyTorch, and OpenCV. Organizations register independently, upload member photos through a web dashboard, and the system automatically trains a custom model, recognizes faces in real time, logs attendance, and blocks spoofing attempts.

---

## Features

- **Multi-tenant** — each organization has fully isolated members, models, and attendance data
- **Real-time face recognition** — live webcam feed in the browser with bounding boxes and confidence scores
- **Auto-training** — model retrains automatically in the background when a new member is added
- **Two-layer liveness detection** — blocks printed photos and video replay attacks
  - Texture analysis (Laplacian variance + gradient + std deviation)
  - Server-side blink detection via OpenCV eye cascade
- **Attendance logging** — per-org timestamped records with 60-second cooldown to prevent duplicate entries
- **Unknown face snapshots** — captures and stores unrecognized faces for review
- **Spoof snapshots** — separately archives detected spoofing attempts
- **JWT authentication** — secure Bearer token auth, 24-hour expiry
- **MVC architecture** — clean separation of controllers, services, models, and core

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Deep Learning | PyTorch · ResNet-18 transfer learning |
| Face Detection | OpenCV Haar Cascade |
| Liveness Detection | OpenCV eye cascade · texture analysis |
| Frontend | Vanilla JS SPA · Chart.js · Bootstrap Icons |
| Database | TinyDB (JSON, multi-table) |
| Auth | JWT (python-jose) · bcrypt |
| Training | CosineAnnealingLR · differential learning rates |

---

## Project Structure

```
face-recognition-api/
│
├── app/
│   ├── controllers/          # HTTP layer — parse request, call service, return response
│   │   ├── auth_controller.py
│   │   ├── member_controller.py
│   │   ├── attendance_controller.py
│   │   ├── recognition_controller.py
│   │   ├── training_controller.py
│   │   └── dashboard_controller.py
│   │
│   ├── services/             # Business logic
│   │   ├── auth_service.py
│   │   ├── member_service.py
│   │   ├── attendance_service.py
│   │   ├── recognition_service.py
│   │   ├── training_service.py
│   │   └── liveness_service.py
│   │
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response schemas
│   │
│   ├── core/
│   │   ├── auth.py           # JWT creation and verification
│   │   ├── database.py       # TinyDB tables
│   │   └── dependencies.py   # FastAPI dependencies, recognizer cache
│   │
│   └── main.py               # App entry point, router registration
│
├── templates/
│   └── dashboard.html        # Single-page dashboard (login + all pages)
│
├── model.py                  # FaceRecognitionModel class + Recognizer inference
├── trainer.py                # Per-org training pipeline
├── realtime.py               # Standalone OpenCV webcam script
│
├── notebooks/
│   └── face_recognition.ipynb
│
├── dataset/                  # Per-org training images  (gitignored)
│   └── {org_id}/
│       └── {member_name}/
│
├── models/                   # Per-org trained weights  (gitignored)
│   └── {org_id}_model.pth
│
├── unknown/                  # Unrecognized face snapshots
├── spoofs/                   # Spoofing attempt snapshots
├── db.json                   # TinyDB database
└── requirements.txt
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/ayanchyaziz123/face-recognition-api.git
cd face-recognition-api
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the server**
```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Quick Start

### 1. Register your organization
On the login screen, click **Register** and create an account for your organization.

### 2. Add members
Go to **Members → Add Member**, enter a name, and upload face photos (10+ clear photos recommended). The model will retrain automatically in the background — a progress banner shows training status.

### 3. Live recognition
Go to **Live Camera**. Look at the camera and **blink twice** to pass liveness verification. Once verified, the system recognizes faces and logs attendance automatically.

### 4. View attendance
Go to **Attendance** to see a full timestamped log with confidence scores. Use the search bar to filter by name.

---

## API Reference

All protected endpoints require the header:
```
Authorization: Bearer <token>
```

### Auth

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/auth/register` | `{org_name, email, password}` | Register a new organization |
| POST | `/auth/login` | `{email, password}` | Login, returns JWT token |
| GET | `/auth/me` | — | Get current org profile |

### Members

| Method | Endpoint | Description |
|---|---|---|
| POST | `/members/add?name=Name` | Upload photos, triggers auto-training |
| GET | `/members/` | List all members |
| DELETE | `/members/{name}` | Remove a member |

### Recognition

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Submit a frame (multipart image), returns faces with liveness |
| GET | `/unknown-list` | List unknown face snapshot filenames |

**Predict response example:**
```json
{
  "face_detected": true,
  "results": [
    {
      "name": "Ayan",
      "greeting": "Hello Ayan!",
      "confidence": 0.963,
      "bbox": { "x": 120, "y": 80, "w": 100, "h": 100 },
      "liveness": {
        "is_live": true,
        "texture_ok": true,
        "texture_variance": 142.5,
        "blink_verified": true,
        "blink_count": 2,
        "eyes_open": true
      }
    }
  ]
}
```

### Attendance

| Method | Endpoint | Description |
|---|---|---|
| GET | `/attendance` | Get all attendance records for the org |
| DELETE | `/attendance` | Clear all attendance records |

### Training

| Method | Endpoint | Description |
|---|---|---|
| POST | `/train` | Start background training |
| GET | `/train/status` | Poll training progress |
| GET | `/reload` | Hot-reload model without server restart |

### System

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |

---

## Model Architecture

ResNet-18 backbone with a custom classification head:

```
ResNet-18 backbone (pretrained ImageNet weights)
  └── Frozen layers 0–6
  └── Unfrozen layer4 + avgpool (fine-tuned at lr=1e-4)

Custom head:
  Flatten → Linear(512, 256) → ReLU → Dropout(0.4) → Linear(256, num_classes)
  (trained at lr=1e-3)
```

**Training config:**
- 30 epochs · CosineAnnealingLR scheduler
- Batch size 16 · Adam optimizer · weight decay 1e-4
- 80/20 train/val split · saves best validation accuracy weights
- Augmentation: horizontal flip, rotation ±15°, color jitter

---

## Liveness Detection

Two independent checks must both pass before a face is accepted as live:

**Layer 1 — Texture analysis (anti-photo)**

| Signal | Threshold | What it catches |
|---|---|---|
| Laplacian variance | > 50 | Flat printed photos score 10–40 |
| Gradient mean | > 8 | Prints lack natural edge complexity |
| Pixel std deviation | > 20 | Displayed screens have uniform regions |

Majority vote: 2 of 3 must pass.

**Layer 2 — Blink detection (anti-video)**

- OpenCV `haarcascade_eye_tree_eyeglasses.xml` detects eyes per frame
- Open → closed transition = one blink registered
- Requires 2 blinks within a 12-second window
- State tracked server-side per organization

---

## Data Isolation (Multi-Tenant)

Each organization is completely isolated:

| Resource | Path |
|---|---|
| Training images | `dataset/{org_id}/{member_name}/` |
| Trained model | `models/{org_id}_model.pth` |
| Attendance records | TinyDB filtered by `org_id` |
| Unknown snapshots | `unknown/{org_id}/` |
| Spoof snapshots | `spoofs/{org_id}/` |

---

## Requirements

- Python 3.10+
- Webcam
- ~2 GB disk space for PyTorch

```
fastapi
uvicorn[standard]
opencv-contrib-python
numpy
python-multipart
torch
torchvision
Pillow
scikit-learn
seaborn
matplotlib
jinja2
aiofiles
tinydb
python-jose[cryptography]
bcrypt
```

---

## Dataset Recommendations

| Photos per person | Expected accuracy |
|---|---|
| 5–10 | ~70% |
| 15–25 | ~85–90% |
| 40+ | ~95%+ |

Use clear, well-lit front-facing photos. Avoid heavy blur or extreme angles. Variety in lighting and expression improves robustness.
