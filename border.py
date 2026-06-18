# A quick sketch to map the lexicon primitives
from google import genai
import json

client = genai.Client()

prompt = """
Give me a clean, tightly scoped list of primitives for a custom robotics vocabulary based on an OmniAddress structure: subject.verb.object.tense.negator.

The robot is a tracked tank chassis running on a Raspberry Pi 5 with a robotic arm, servos, an ultrasonic sensor, a camera using YOLO-style tracking, and conversational audio.

Provide 10-15 solid options for:
- subjects (who/what initiates)
- verbs (actions/states)
- objects (targets of action or vision)

Keep them strictly lowercase, single-word primitives (e.g., 'robot', 'adam', 'chair', 'stalled', 'steered'). Return as raw JSON with keys: 'subjects', 'verbs', 'objects'.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
    config={"response_mime_type": "application/json"}
)

print(json.dumps(json.loads(response.text), indent=2))
