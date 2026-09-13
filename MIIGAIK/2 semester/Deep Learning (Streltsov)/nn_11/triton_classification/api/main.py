import base64
import io
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from pydantic import BaseModel
from PIL import Image
import httpx

# Адрес Triton и параметры модели
TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
INPUT_NAME = os.getenv("INPUT_NAME", "input")
# ВАЖНО: имя выхода должно совпадать с именем тензора в ONNX / логе Triton
OUTPUT_NAME = os.getenv("OUTPUT_NAME", "Identity:0")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "32"))

# Классы из датасета цветов
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

app = FastAPI(title="Image Classification Gateway", version="1.0.0")


class PredictRequest(BaseModel):
    image_base64: Optional[str] = None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image, dtype=np.float32)
    # [N, H, W, C] = [1, 224, 224, 3]
    arr = np.expand_dims(arr, axis=0)
    return arr


def decode_base64_image(image_base64: str) -> bytes:
    # Поддержка формата "data:image/jpeg;base64,...."
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    try:
        return base64.b64decode(image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc


async def triton_infer(batch: np.ndarray) -> np.ndarray:
    # Формируем HTTP v2 запрос в Triton
    payload = {
        "inputs": [
            {
                "name": INPUT_NAME,
                "shape": list(batch.shape),
                "datatype": "FP32",
                "data": batch.reshape(-1).tolist(),
            }
        ],
        "outputs": [{"name": OUTPUT_NAME}],
    }

    url = f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/infer"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Triton inference request failed: {exc}",
            ) from exc

        result = response.json()

    outputs = result.get("outputs", [])
    if not outputs:
        raise HTTPException(status_code=502, detail="Empty response from Triton")

    data = outputs[0].get("data")
    if data is None:
        raise HTTPException(status_code=502, detail="Triton output has no data")

    return np.array(data, dtype=np.float32).reshape(batch.shape[0], -1)


def format_prediction(logits: np.ndarray) -> dict:
    probs = logits[0].astype(float)
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASSES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
    }


@app.get("/health")
async def health():
    model_url = f"{TRITON_URL}/v2/models/{MODEL_NAME}"
    metrics_url = f"{TRITON_URL}/metrics"
    status = {
        "api": "ok",
        "triton": "down",
        "model": MODEL_NAME,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            model_resp = await client.get(model_url)
            metrics_resp = await client.get(metrics_url)

        status["triton"] = "ok" if model_resp.status_code == 200 else "down"
        status["model_ready"] = model_resp.status_code == 200
        status["metrics_available"] = metrics_resp.status_code == 200
    except Exception:
        status["model_ready"] = False
        status["metrics_available"] = False

    return status


@app.get("/classes")
async def classes():
    return {"model": MODEL_NAME, "classes": CLASSES, "count": len(CLASSES)}


# @app.post("/predict")
# async def predict(
#     file: Optional[UploadFile] = File(default=None),
#     payload: Optional[PredictRequest] = Body(default=None),
# ):
#     # Должен быть либо файл, либо корректный base64
#     if file is None and (payload is None or not payload.image_base64):
#         raise HTTPException(
#             status_code=400,
#             detail="Provide either an image file or image_base64",
#         )
#
#     if file is not None:
#         image_bytes = await file.read()
#     else:
#         image_bytes = decode_base64_image(payload.image_base64)
#
#     batch = preprocess_image(image_bytes)
#     logits = await triton_infer(batch)
#     return format_prediction(logits)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Image file is required",
        )

    image_bytes = await file.read()
    batch = preprocess_image(image_bytes)
    logits = await triton_infer(batch)
    return format_prediction(logits)