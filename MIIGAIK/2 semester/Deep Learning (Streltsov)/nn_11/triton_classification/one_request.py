import requests
import numpy as np
from PIL import Image
import io
import base64

API_URL = "http://localhost:8080/predict"

def generate_test_image_base64() -> str:
    img = Image.fromarray(
        np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

payload = {"image_base64": generate_test_image_base64()}

resp = requests.post(API_URL, json=payload)
print("Status:", resp.status_code)
print("Body:", resp.text)