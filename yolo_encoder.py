"""
yolo_encoder.py — the VISION encoder: YOLO detection → 6-field address.

The sibling of OmniAddress on the opposite side of the address bus:
    OmniAddress : English  → address   (intent / hands)
    yolo_encoder: detection → address   (perception / eyes)
Both land in the SAME space, so vision and language unify natively (the hex thesis),
and both decode to English through verbalizer.gloss() — which is how a TEXT-ONLY
model "sees": the scene arrives as language, not pixels.

WHAT RIDES THE BUS (decided 2026-06-04): only the SYMBOLIC content of a detection.
  • subject     = the detected class (what it is)
  • destination = the relative ZONE (left/right/ahead) — position is a RELATION, so
                  it's address-worthy ("on the left"), not a measurement.
  • verb=detect, object=none, tense=now, negator=true  (a live, holding observation)
WHAT DOES NOT: distance/area/confidence are MEASUREMENTS → telemetry, like odometry
magnitude ("an inch"). They never become schema fields. (Ultrasonic/IR never reach
here at all — they're reflex + occupancy-grid, below cognition. See CLAUDE.md.)

Duck-typed on the Detection fields (.class_name, .relative_x, .confidence) so this
encodes offline with a stub — no ultralytics, no camera. The live PerceptionEngine
(xiaor/signalbot_perception.py) hands us the real Detections unchanged.
"""

from typing import List
from verbalizer import gloss

# Horizontal zone cuts on relative_x (0.0 = far left, 1.0 = far right).
# Center band → "ahead" (the thing is in front of the tank).
LEFT_EDGE = 1 / 3
RIGHT_EDGE = 2 / 3


def _zone(relative_x: float) -> str:
    """Relative-x → a relational position the verbalizer renders ('on the left'/'ahead')."""
    if relative_x < LEFT_EDGE:
        return "left"
    if relative_x > RIGHT_EDGE:
        return "right"
    return "ahead"


def _subject(class_name: str) -> str:
    """COCO label → an address token. Spaces → '_' (the address is dot-delimited);
    the verbalizer turns it back ('cell_phone' → 'a cell phone')."""
    s = class_name.strip().lower().replace(" ", "_")
    return s or "object"


def detection_to_address(det) -> str:
    """One Detection → a 6-field perception address string."""
    return f"{_subject(det.class_name)}.detect.none.{_zone(det.relative_x)}.now.true"


def encode_detections(dets, min_conf: float = 0.0) -> List[str]:
    """Many Detections → addresses. min_conf filters weak hits (noise off the bus)."""
    return [detection_to_address(d) for d in dets if getattr(d, "confidence", 1.0) >= min_conf]


def scene_gloss(dets, min_conf: float = 0.25) -> str:
    """THE PERCEPTION-FEED: detections → addresses → English scene for the LLM prompt.

    This is the string you inject into a text-only model's context so it can reason
    about what the camera sees. Empty string when nothing is detected (say nothing
    rather than narrate a void)."""
    lines = [gloss(addr) for addr in encode_detections(dets, min_conf)]
    return ". ".join(lines) + "." if lines else ""


# ═══════════════════════════════════════════════════════════════════
# SCENE PROVIDER — the camera stays decoupled from the brain.
# The robot side registers its detection source ONCE:
#     set_scene_provider(perception_engine.latest)
# Anything that builds an LLM prompt then calls current_scene() to ground the
# model in live sight. No provider (e.g. the web app with no camera) → "" → the
# prompt block is simply skipped. One helper, two consumers (chat turn + tick).
# ═══════════════════════════════════════════════════════════════════

_scene_provider = None   # callable() -> list[Detection]
# Fix #2 (2026-06-11 audit): the remembered scene was a single module global,
# so one web user's uploaded photo leaked into EVERY other user's next text
# turn as "their" live sight. Now keyed per user; the CLI / single-user path
# just uses the default key.
_last_scenes: dict = {}  # user → last uploaded photo's glossed scene


def set_scene_provider(fn) -> None:
    """Register the live detection source (e.g. PerceptionEngine.latest). None disables."""
    global _scene_provider
    _scene_provider = fn


def remember_scene(text: str, user: str = "_default") -> None:
    """Cache the latest uploaded photo's scene so follow-up TEXT turns still 'see' it.
    Without this, sight lasts only the single turn the image bytes are attached — a
    follow-up like "what is this?" arrives blind. A live provider (robot camera) always
    overrides this; this is purely the no-camera phone fallback. Empty overwrites empty:
    a new photo is a new view, so a bad/empty frame honestly clears the last sight.
    Per-user keyed: one person's photo must never become someone else's sight."""
    _last_scenes[user] = text or ""


def current_scene(min_conf: float = 0.25, user: str = "_default") -> str:
    """Glossed scene: live provider if registered, else the last uploaded photo.
    Perception must never crash the talk lane — a blind turn beats a dead one."""
    if _scene_provider is None:
        return _last_scenes.get(user, "")  # phone path: this user's last upload only
    try:
        dets = _scene_provider() or []
    except Exception:
        return ""
    return scene_gloss(dets, min_conf)


if __name__ == "__main__":
    # eyeball with stub detections — `python3 yolo_encoder.py`
    from types import SimpleNamespace as D
    dets = [
        D(class_name="person", confidence=0.91, relative_x=0.20),
        D(class_name="chair",  confidence=0.77, relative_x=0.85),
        D(class_name="cell phone", confidence=0.66, relative_x=0.50),
        D(class_name="bird",   confidence=0.10, relative_x=0.5),  # below min_conf → dropped
    ]
    for d in dets:
        print(f"  {d.class_name:11s} x={d.relative_x} → {detection_to_address(d)}")
    print("\n  scene →", scene_gloss(dets))
