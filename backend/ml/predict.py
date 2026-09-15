"""AgriSmart AI — crop disease inference.

Loads the trained EfficientNet-B0 checkpoint once (lazily, on first request)
and exposes predict_bytes() for image bytes coming from an upload.
"""

from __future__ import annotations

import io
import json
import os
import threading
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ML_DIR, "models")
CHECKPOINT = os.path.join(MODEL_DIR, "crop_disease_efficientnet_b0.pth")

# ImageNet statistics — the checkpoint was fine-tuned from ImageNet weights.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_lock = threading.Lock()
_state: dict[str, Any] = {}


def _build(num_classes: int) -> torch.nn.Module:
    net = models.efficientnet_b0(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return net


def _load() -> dict[str, Any]:
    """Load the checkpoint once. Safe to call from multiple threads/requests."""
    if _state:
        return _state
    with _lock:
        if _state:
            return _state

        if not os.path.exists(CHECKPOINT):
            raise FileNotFoundError(
                f"Model checkpoint not found at {CHECKPOINT}. "
                "Make sure backend/ml/models/crop_disease_efficientnet_b0.pth is present."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The checkpoint was saved from a CUDA run, so map_location is required
        # for it to load on a CPU-only server.
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

        classes = ckpt.get("classes")
        if classes is None:
            with open(os.path.join(MODEL_DIR, "classes.json")) as fh:
                classes = json.load(fh)

        size = ckpt.get("image_size", 224)
        if isinstance(size, (list, tuple)):
            size = int(size[0])

        net = _build(len(classes))
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval().to(device)

        tf = transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        _state.update(
            model=net,
            classes=classes,
            transform=tf,
            device=device,
            image_size=size,
            architecture=ckpt.get("architecture", "efficientnet_b0"),
        )
    return _state


def split_label(raw: str) -> tuple[str, str, bool]:
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight', is_healthy)."""
    plant, _, condition = raw.partition("___")
    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip() or "Unknown"
    healthy = condition.lower() == "healthy"
    return plant, condition, healthy


def predict_bytes(data: bytes, top_k: int = 5) -> dict[str, Any]:
    """Classify raw image bytes from an upload."""
    st = _load()
    with Image.open(io.BytesIO(data)) as im:
        image = im.convert("RGB")
        tensor = st["transform"](image).unsqueeze(0).to(st["device"])

    with torch.no_grad():
        probs = F.softmax(st["model"](tensor), dim=1)[0].cpu()

    k = min(top_k, len(st["classes"]))
    scores, indices = torch.topk(probs, k)

    ranked = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        raw = st["classes"][idx]
        plant, condition, healthy = split_label(raw)
        ranked.append({
            "label": raw,
            "plant": plant,
            "condition": condition,
            "healthy": healthy,
            "confidence": round(float(score), 6),
        })

    best = ranked[0]
    return {
        "class": best["label"],
        "plant": best["plant"],
        "condition": best["condition"],
        "healthy": best["healthy"],
        "confidence": best["confidence"],
        "topK": ranked,
        "model": st["architecture"],
    }


def warmup() -> None:
    """Load weights ahead of the first request so it doesn't pay the cost."""
    _load()
