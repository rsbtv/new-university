from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
import os
import ast

app = FastAPI(title="Flower Classification API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "best_classification_model.h5")
MODEL_INFO_PATH = os.getenv("MODEL_INFO_PATH", "model_info.txt")

model = None
class_names = None
img_size = None
needs_flatten = False
preprocessing = "standard"


def load_model_info(path: str):
    global class_names, img_size, needs_flatten, preprocessing
    if not os.path.exists(path):
        raise FileNotFoundError(f"model_info.txt not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().splitlines()

    info = {}
    for line in content:
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

    img_size = ast.literal_eval(info.get("IMG SIZE", "(64, 64)"))
    needs_flatten = info.get("FLATTEN", "False") == "True"
    preprocessing = info.get("PREPROCESSING", "standard")
    class_names = ast.literal_eval(info.get("КЛАССЫ", "[]"))


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(img_size)
    arr = np.array(image).astype("float32")

    if preprocessing == "imagenet":
        arr = arr / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
    elif preprocessing == "tanh":
        arr = (arr / 127.5) - 1.0
    else:
        arr = arr / 255.0

    if needs_flatten:
        arr = arr.reshape(-1)

    arr = np.expand_dims(arr, axis=0)
    return arr


@app.on_event("startup")
def startup_event():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    load_model_info(MODEL_INFO_PATH)
    model = keras.models.load_model(MODEL_PATH, compile=False)


@app.get("/")
def root():
    return {
        "message": "Flower Classification API is running",
        "model_path": MODEL_PATH,
        "img_size": img_size,
        "needs_flatten": needs_flatten,
        "preprocessing": preprocessing,
        "classes": class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Upload an image: jpg, jpeg, png, or webp")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(image)
        probs = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)

        return JSONResponse({
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "probabilities": {
                class_names[i] if class_names and i < len(class_names) else str(i): float(probs[i])
                for i in range(len(probs))
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
