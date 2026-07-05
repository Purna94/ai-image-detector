import io
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import pipeline

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MODEL_NAME = "umm-maybe/AI-image-detector"

app = FastAPI(title="AI Image Detector")

BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

classifier: pipeline = None


@app.on_event("startup")
def load_model():
    global classifier
    classifier = pipeline("image-classification", model=MODEL_NAME)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": classifier is not None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    results = classifier(image, top_k=2)
    predictions = [
        {
            "label": r["label"],
            "confidence": round(r["score"] * 100, 2),
        }
        for r in results
    ]
    top = predictions[0] if predictions else {}
    return {"predictions": predictions, "top_label": top.get("label"), "top_confidence": top.get("confidence")}
