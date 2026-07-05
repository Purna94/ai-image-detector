# AI Image Detector

[![CI](https://github.com/your-username/ai-image-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/ai-image-detector/actions/workflows/ci.yml)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Upload an image and instantly find out if it's **AI-Generated** or **Real** — powered by a Hugging Face transformer model.

## Features

- Real-time classification using `umm-maybe/AI-image-detector`
- Drag-and-drop interface with live image preview
- Confidence breakdown for both classes
- In-memory image processing (no disk writes)
- Model caching at startup (no per-request download)
- Containerized with multi-stage Docker build
- CI/CD via GitHub Actions

## Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-image-detector.git
cd ai-image-detector

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Deployment

### Docker

```bash
docker build -t ai-image-detector .
docker run -p 8000:8000 ai-image-detector
```

### Render / Railway / Hugging Face Spaces

1. Push this repository to GitHub.
2. On your chosen platform, create a new service from the GitHub repo.
3. Set the **Start Command** to `uvicorn app:app --host 0.0.0.0 --port 8000`.
4. Use the included `Dockerfile` for containerized deployment.

## API

### `GET /health`

Returns the service health status.

### `POST /api/predict`

Upload an image file (PNG, JPG, JPEG, WEBP) and get classification results.

```json
{
  "predictions": [
    { "label": "Real", "confidence": 87.32 },
    { "label": "AI-Generated", "confidence": 12.68 }
  ],
  "top_label": "Real",
  "top_confidence": 87.32
}
```

## License

MIT
