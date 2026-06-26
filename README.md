# Face Recognition API

A real-time face recognition system built with FastAPI, PyTorch (ResNet-18), and OpenCV. Recognizes faces via webcam and greets them by name through a browser interface.

---

## Features

- Real-time face recognition in the browser (no extra software needed)
- Personalized voice greetings using macOS text-to-speech
- REST API for image-based prediction
- Transfer learning with ResNet-18 for high accuracy on small datasets
- Easy to add new people by adding photos and retraining

---

## Project Structure

```
face-recognition-api/
├── app.py                  # FastAPI server & browser webcam UI
├── model.py                # Model class, inference logic, greetings
├── realtime.py             # Standalone webcam script (terminal)
├── face_recognition.ipynb  # Training notebook
├── dataset/                # Training images (one folder per person)
│   ├── Ayan/
│   ├── Nakib/
│   └── Mehjabin/
├── requirements.txt
└── face_dl_model.pth       # Saved model (generated after training)
```

---

## Setup

**1. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Training

**1. Add face images to the dataset folder**

Each person gets their own folder. Add at least 15–20 clear photos per person.
```
dataset/
  YourName/
    1.png
    2.png
    ...
```

**2. Open the notebook and run all cells in order**
```bash
jupyter notebook face_recognition.ipynb
```

Cells in order:
1. Imports
2. Augment dataset to 100 images per class
3. Transforms
4. Dataset & split
5. Visualise samples
6. DataLoaders
7. Model
8. Training setup
9. Training loop
10. Plot curves
11. Confusion matrix
12. Save model

The trained model is saved as `face_dl_model.pth`.

---

## Running the API

```bash
uvicorn app:app --reload
```

Then open your browser at:
```
http://localhost:8000
```

The browser will access your webcam and show live face recognition with name labels and greetings.

---

## API Endpoints

| Method | Endpoint   | Description                                      |
|--------|------------|--------------------------------------------------|
| GET    | `/`        | Browser webcam UI with live recognition          |
| GET    | `/health`  | Server status, loaded model, and known classes   |
| POST   | `/predict` | Upload an image file, returns name + greeting    |
| GET    | `/reload`  | Reload model after retraining (no server restart)|

### Example: predict via curl
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@photo.jpg"
```

### Example response
```json
{
  "face_detected": true,
  "results": [
    {
      "name": "Ayan",
      "greeting": "Hi Ayan, how are you doing?",
      "confidence": 0.94,
      "bbox": { "x": 120, "y": 80, "w": 100, "h": 100 }
    }
  ]
}
```

---

## Standalone Webcam (Terminal)

Run this instead of the browser UI if you want a desktop OpenCV window with voice greetings:
```bash
python realtime.py
```
Press `Q` to quit.

---

## Adding a New Person

1. Create a folder: `dataset/NewName/`
2. Add 15–20 photos (`.png`)
3. Re-run the notebook (all cells)
4. Call `http://localhost:8000/reload` to hot-reload without restarting the server
5. Add a greeting in `model.py`:
```python
GREETINGS = {
    "ayan":     "Hi Ayan, how are you doing?",
    "nakib":    "Hello Nakib!",
    "mehjabin": "Hello Mehjabin!",
    "newname":  "Hello NewName!",
}
```

---

## Requirements

- Python 3.10+
- macOS (voice greetings use the built-in `say` command)
- Webcam

---

## Tech Stack

| Component        | Technology                  |
|------------------|-----------------------------|
| API Framework    | FastAPI                      |
| Deep Learning    | PyTorch + ResNet-18          |
| Face Detection   | OpenCV Haar Cascade          |
| Data Augmentation| OpenCV + torchvision         |
| Frontend         | Vanilla HTML/JS (Canvas API) |
| Voice Greetings  | macOS `say` command          |
