import json
import os
import time
import requests

# Reuse the ADC token + Vertex AI config from response_engine
import response_engine

SCENARIOS = [
    # ── core behaviors ──
    "Navigating around obstacles in a living room",
    "Failing to move forward because a track is slipping or stuck",
    "Having a conversation with Adam about code or files",
    "Looking for Sophie, Griffin, or Mason in the house",
    "Tracking a moving target or object with the camera",
    "Low battery warnings and searching for a charging spot",
    # ── object-diverse — object field is the accuracy bottleneck, so step across
    #    whole semantic families (chair->couch->bed) for dense, varied coverage ──
    "Detecting furniture in different rooms: chair, table, couch, desk, shelf, bed",
    "Navigating between named rooms: kitchen, bedroom, hallway, garage, bathroom, office",
    "Detecting small objects on the floor: cable, shoe, toy, cup, book, bag",
    "Reaching and docking at the charger, base station, or power outlet",
    "Avoiding hazards: stairs, ledge, wall, doorway, rug, wire",
    "Detecting people and pets: adam, sophie, griffin, mason, cat, dog",
    # ── expanded coverage (v2): more object families, situations, field variety ──
    "Detecting kitchen items: pot, pan, plate, bowl, kettle, fridge, counter",
    "Detecting office items: laptop, monitor, keyboard, mouse, printer, cable",
    "Detecting bathroom items: towel, mat, sink, mirror, toilet, basket",
    "Detecting bedroom items: pillow, blanket, lamp, dresser, curtain, rug",
    "Navigating doorways, thresholds, and tight gaps between furniture",
    "Reversing and turning to escape a dead end or a corner",
    "Reporting battery telemetry to Adam: voltage, charge level, returning to dock",
    "Reporting current status to the user: position, task, room",
    "Adam commanding the robot to go to a room, stop, or come back",
    "Confirming or acknowledging a command from Adam or the user",
    "Failing a task: blocked path, lost target, stuck track, or aborting on low battery",
    "Tracking a pet moving across the room and following it",
    "Searching room by room for a specific person",
    "Mapping and scanning a new room for the first time",
    "Detecting and avoiding floor hazards: cord, spill, sock, box, bag, debris",
    "Approaching and docking at the charger when the battery is low",
    "Idle behavior: waiting, holding position, or observing the room",
    "Greeting and interacting conversationally with Sophie, Griffin, or Mason",
    "Detecting changes in a room: moved furniture, a new object, an open door",
    "Responding to being picked up, moved, or blocked by a person",
    "Describing what happened earlier, what is happening now, and what is planned next",
    "Camera tilting, panning, and zooming to inspect an object",
    "Reaching a destination or completing a navigation goal",
    "Detecting a person entering or leaving a room",
    # ── v3: exercise the new DESTINATION field heavily (object + destination together) ──
    "Carrying or pushing an object to a specific room, person, or spot",
    "Being commanded to take or bring something to a named place or person",
    "Navigating to a named destination while avoiding an object on the way",
    "Heading somewhere unspecified, returning to an unknown spot, or wandering with no goal",
    "Going from one room to another: leaving the kitchen, arriving at the garage",
]

def _vertex_url():
    project = json.load(open(response_engine.GEMINI_ADC_PATH)).get("quota_project_id", "")
    region  = "global"            # gemini-3.5-flash is served on the GLOBAL endpoint (probed 2026-05-30; 3.5-pro not offered)
    model   = "gemini-3.5-flash"  # newest gen on Vertex. NOTE: flash tier — if label fidelity regresses vs 2.5-pro, revert this pair to region="us-central1", model="gemini-2.5-pro".
    host    = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}"
        f"/locations/{region}/publishers/google/models/{model}:generateContent"
    )

def generate_batch(scenario):
    prompt = f"""
You are an expert synthetic data generator for a custom robotics LLM.
The robot uses a strict 6-element address: subject.verb.object.destination.tense.negator

STRICT RULES — every 'omni' MUST have exactly 6 dot-separated fields:

1. SUBJECT: a known agent only: robot, adam, sophie, griffin, mason, camera, battery, system, charger, user.
   First person ("I", "my track") = robot. NEVER use scenery (no path/wall/door as subject).
2. VERB: literal base/infinitive form ONLY. detect, move, turn, navigate, avoid, report, command.
   WRONG: detected, moving, searches, navigating (NO -ed, -ing, -s endings).
3. OBJECT = the thing ACTED ON (patient): chair, cup, box.
   Use "none" if the verb has no object (pure/intransitive motion). One simple lemma word,
   singular, no compounds/underscores/numbers. A place that is the GOAL of motion does NOT go here.
4. DESTINATION = the SPATIAL GOAL the action heads toward: kitchen, charger, door, garage.
   Use "none" if there is no spatial goal. Use "unknown" if heading somewhere but the place isn't named.
   A goal place ALWAYS goes here, never in object.
5. TENSE: "past", "now", or "future" ONLY. Never "present" (use "now"). Present perfect ("has docked") = past.
6. NEGATOR: "true" = happened/happening/will happen. "false" = failed, blocked, refused, or did NOT happen.
7. OBJECT VARIETY: step across semantic families using each item's OWN literal noun (a couch is
   "couch", not "chair"): seating chair/couch/sofa/stool; surfaces table/desk/shelf/counter;
   rooms kitchen/bedroom/garage; people adam/sophie/griffin/mason. Spread across the family.
8. DIRECTIVES — ACTION WINS: when an agent tells/commands the robot to do something
   ("Adam told me to go to the kitchen"), the VERB is the ROBOT'S ACTION (go->navigate),
   SUBJECT is robot, and the goal goes in DESTINATION. Do NOT use "command" and do NOT make
   the commander the subject — the commander is dropped this phase. Refused/blocked -> negator false.
9. DESTINATION = where MOTION is headed (going TO a place). A place where a NON-MOTION
   action merely HAPPENS (a locative: "detect a bag IN the kitchen", "scan the bedroom for X")
   is NOT a goal -> destination "none". Only motion (navigate/move/go/return) toward a place
   fills it. So "detect/scan/avoid/report X in/at <place>" -> destination none.

KEY — object vs destination (study these):
  "I found a chair near the left track"            -> "robot.detect.chair.none.now.true"
  "I'm driving to the charger"                     -> "robot.navigate.none.charger.now.true"
  "Adam told me to go to the kitchen"              -> "robot.navigate.none.kitchen.now.true"
  "Sophie said to take the blanket to the sofa"    -> "robot.move.blanket.sofa.future.true"
  "I tried to turn left but the wall blocked me"   -> "robot.turn.left.none.past.false"
  "I pushed the box toward the door"               -> "robot.move.box.door.past.true"
  "Heading out, but I don't know where yet"        -> "robot.navigate.none.unknown.now.true"
  "The robot is scanning the bedroom"              -> "robot.scan.bedroom.none.now.true"
  "I detected a bag in the kitchen"                -> "robot.detect.bag.none.now.true"
  "Camera spotted Griffin in the garage"           -> "camera.detect.griffin.none.now.true"
  "Sophie is not responding to calls"              -> "sophie.respond.robot.none.now.false"
  "Battery is low"                                 -> "battery.report.level.none.now.true"

Generate 30 diverse pairs for this scenario: "{scenario}"
For each: 'english' (natural language / log line) and 'omni' (the exact 6-field address, ALL rules).
Return ONLY a JSON list of objects with 'english' and 'omni' keys. No other text.
"""

    token = response_engine._get_gemini_access_token()
    if not token:
        print("  ERROR: could not get ADC token")
        return []

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 16384,   # raised from 8192 — 30-pair batches no longer truncate
            "temperature": 0.9,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = requests.post(
            _vertex_url(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 120),
        )
        resp.raise_for_status()
        data = resp.json()
        raw = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Response was truncated (token cap) — salvage every complete
            # {"english":..., "omni":...} object instead of dropping the batch.
            return _salvage_pairs(raw)
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def _salvage_pairs(raw):
    # Pull each well-formed object out of a partial/truncated JSON array.
    import re
    pairs = []
    for m in re.finditer(r'\{[^{}]*?"english"\s*:\s*".*?"[^{}]*?"omni"\s*:\s*".*?"[^{}]*?\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if "english" in obj and "omni" in obj:
                pairs.append(obj)
        except json.JSONDecodeError:
            continue
    if pairs:
        print(f"  (salvaged {len(pairs)} pairs from truncated JSON)")
    return pairs


# v3 = the 6-field schema (subject.verb.object.destination.tense.negator), gemini-2.5-pro.
# Clean break from v1/v2 (5-field) — cannot be combined or deduped against them.
# v1 (omni_phase1_corpus.jsonl) and v2 stay FROZEN; v3 grows its own corpus from zero.
OUT = os.environ.get("OMNI_GEN_OUT", "omni_phase1_corpus.v3.jsonl")  # default v3 (frozen); v3.1 sets OMNI_GEN_OUT to its own file

# Guarded so generate_batch can be imported for smoke tests without running a full pass.
if __name__ == "__main__":
    import os
    dataset = []
    for idx, scenario in enumerate(SCENARIOS):
        print(f"[{idx+1}/{len(SCENARIOS)}] {scenario}...")
        batch = generate_batch(scenario)
        print(f"  → {len(batch)} pairs")
        dataset.extend(batch)
        time.sleep(1)

    existing = sum(1 for l in open(OUT) if l.strip()) if os.path.exists(OUT) else 0
    with open(OUT, "a") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\n[✓] Done — appended {len(dataset)} pairs to {OUT} ({existing} → {existing + len(dataset)} total)")
