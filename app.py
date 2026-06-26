from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from model import Recognizer

app        = FastAPI(title="Face Recognition API")
recognizer = Recognizer()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Face Recognition</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f0f0f;
      color: #fff;
      font-family: 'Segoe UI', Arial, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      gap: 16px;
    }
    h1 { font-size: 1.5rem; letter-spacing: 1px; color: #eee; }
    #canvas { border-radius: 12px; box-shadow: 0 0 30px rgba(0,200,100,0.2); max-width: 95vw; }
    #status {
      font-size: 0.9rem;
      color: #888;
      background: #1a1a1a;
      padding: 6px 16px;
      border-radius: 20px;
    }
    video { display: none; }
  </style>
</head>
<body>
  <h1>Real-Time Face Recognition</h1>
  <video id="video" autoplay playsinline></video>
  <canvas id="canvas"></canvas>
  <div id="status">Starting camera...</div>

  <script>
    const video  = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx    = canvas.getContext('2d');
    const status = document.getElementById('status');

    let latestResults = [];
    let busy = false;

    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          canvas.width  = video.videoWidth;
          canvas.height = video.videoHeight;
          status.textContent = 'Camera ready — looking for faces...';
          requestAnimationFrame(draw);
          setInterval(sendFrame, 600);
        };
      })
      .catch(err => status.textContent = 'Camera error: ' + err.message);

    // Draw loop
    function draw() {
      ctx.drawImage(video, 0, 0);

      latestResults.forEach(r => {
        const { x, y, w, h } = r.bbox;
        const known = r.name !== 'Unknown';
        const color = known ? '#00e676' : '#ff1744';

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth   = 3;
        ctx.strokeRect(x, y, w, h);

        // Name tag background
        const tag = `${r.name}  ${(r.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 16px Segoe UI, Arial';
        const tw = ctx.measureText(tag).width;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 28, tw + 12, 24);
        ctx.fillStyle = '#000';
        ctx.fillText(tag, x + 6, y - 10);

        // Greeting below box
        ctx.fillStyle = color;
        ctx.font = '14px Segoe UI, Arial';
        ctx.fillText(r.greeting, x, y + h + 20);
      });

      requestAnimationFrame(draw);
    }

    // Send frame to /predict
    function sendFrame() {
      if (busy) return;
      busy = true;
      canvas.toBlob(blob => {
        const form = new FormData();
        form.append('file', blob, 'frame.jpg');
        fetch('/predict', { method: 'POST', body: form })
          .then(r => r.json())
          .then(data => {
            latestResults = data.results || [];
            status.textContent = data.face_detected
              ? 'Detected: ' + data.results.map(r => r.name).join(', ')
              : 'No face detected';
          })
          .catch(() => {})
          .finally(() => { busy = false; });
      }, 'image/jpeg', 0.8);
    }
  </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": recognizer.model is not None,
        "classes":      recognizer.classes,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        return recognizer.predict(await file.read())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/reload")
def reload():
    recognizer.load()
    if recognizer.model is None:
        raise HTTPException(503, "Model not found. Train in the notebook first.")
    return {"message": "Model reloaded.", "classes": recognizer.classes}
