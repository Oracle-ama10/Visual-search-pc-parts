"""สกัด Feature Vector 512 มิติจากภาพด้วย OpenCLIP ViT-B/32 (กิจกรรมที่ 2)
ตั้ง FAKE_EMBED=1 เพื่อทดสอบระบบโดยไม่ต้องโหลดโมเดลจริง (ใช้เฉพาะตอนเขียนเทสต์)"""
import os, hashlib
import numpy as np
from PIL import Image

_state = {}

def _load():
    if "model" not in _state:
        import torch, open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        _state.update(model=model.to(device).eval(), preprocess=preprocess, device=device, torch=torch)
    return _state

def embed_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    if os.getenv("FAKE_EMBED") == "1":
        h = hashlib.sha256(img.tobytes()).digest()
        v = np.random.default_rng(int.from_bytes(h[:8], "big")).standard_normal(512).astype(np.float32)
    else:
        s = _load(); torch = s["torch"]
        x = s["preprocess"](img).unsqueeze(0).to(s["device"])
        with torch.no_grad():
            v = s["model"].encode_image(x)[0].cpu().numpy().astype(np.float32)
    return v / np.linalg.norm(v)
