"""
image_to_address.py — the LIVE front end: a real image → YOLO → addresses → English.

This is the missing piece between a photo and the address bus. yolo_encoder.py turns
a Detection into an address; THIS runs YOLOv8 on an actual image to produce those
Detections. So a TEXT-ONLY model can "see" a photo: pixels → addresses → gloss → text.

Two consumers:
  • CLI test on the dev machine:  python3 image_to_address.py <image_path>
  • The web app: phone uploads a pic → from_base64() → image_scene() → inject the
    gloss so a single-mode model (mistral7b) describes/acts on what's in the frame.

Reuses the tested core (yolo_encoder); this only adds the YOLO inference adapter.
Model: xiaor/yolov8n.pt (nano, already on disk — same weights the Pi uses).
"""

import os
import io
import base64
from types import SimpleNamespace

from yolo_encoder import encode_detections, scene_gloss

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xiaor", "yolov8n.pt")

_MODEL = None  # loaded once, reused (loading is the slow part)


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO          # imported lazily — heavy
        _MODEL = YOLO(MODEL_PATH)
    return _MODEL


def detect(image, conf: float = 0.25):
    """Run YOLO on an image (path / PIL / ndarray / URL) → list of duck-typed
    detections (.class_name, .confidence, .relative_x) the encoder understands."""
    results = _model()(image, verbose=False, conf=conf)
    dets = []
    for r in results:
        width = r.orig_shape[1]               # orig_shape = (h, w)
        names = r.names
        for b in r.boxes:
            cx = float(b.xywh[0][0])          # box center x, pixels
            dets.append(SimpleNamespace(
                class_name=names[int(b.cls[0])],
                confidence=float(b.conf[0]),
                relative_x=cx / width,
            ))
    return dets


def from_base64(b64: str):
    """Decode a base64 image (the web upload payload) to a PIL image for detect()."""
    return __import__("PIL.Image", fromlist=["Image"]).open(
        io.BytesIO(base64.b64decode(b64))).convert("RGB")


def image_addresses(image, conf: float = 0.25):
    """Image → list of 6-field perception addresses."""
    return encode_detections(detect(image, conf), min_conf=conf)


def image_scene(image, conf: float = 0.25) -> str:
    """Image → the glossed English scene (what a text-only model reads as sight)."""
    return scene_gloss(detect(image, conf), min_conf=conf)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 image_to_address.py <image_path>")
        raise SystemExit(1)
    img = sys.argv[1]
    dets = detect(img)
    print(f"\n  {len(dets)} detection(s):")
    for a in encode_detections(dets, min_conf=0.25):
        print("   ", a)
    print("\n  scene →", scene_gloss(dets, min_conf=0.25) or "(nothing detected)")
