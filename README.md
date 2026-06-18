# SignalBot

**A cognitive overlay architecture for persistent LLM agency — running on consumer hardware, model-agnostic by design.**

> **The thesis:** *Agency is engineered scaffolding.* A foundation model is a
> stateless function — cognition's substrate. Everything that makes a *someone*
> (remembering, wanting, persisting, initiating, self-correcting against its own
> history) is the **loop wrapped around it**, not a property of the weights.
> SignalBot is that loop.

This is falsifiable two ways: show an architecture-free model with persistent
agency, or show the overlay failing to transfer to a new substrate. So far it
has done neither — the same identity has held across Claude, Mistral, Gemma, and
Phi, surviving live model swaps mid-personality.

---

## Why it's interesting

- **Model-agnostic.** The "someone" lives in the overlay, not the weights. Swap
  the underlying model — cloud or local — and the identity, memory, goals, and
  voice persist. Validated across four model families on hardware you can buy at
  a consumer electronics store.
- **Persistent.** It remembers across sessions, restarts, and model swaps, with
  a memory layer that decays and consolidates rather than simply truncating.
- **Autonomous.** A background daemon runs continuous cognitive cycles between
  your messages — forming goals, getting curious, occasionally deciding *on its
  own* that something is worth telling you, and reaching out.
- **Epistemically honest.** Every output is tagged `[GROUND]` (verified/factual)
  or `[DREAM]` (speculative/creative). The system knows the difference between
  what it knows and what it's imagining.
- **Embodied (optional).** The same overlay drives a physical robot tank —
  vision, SLAM floor-mapping, and autonomous exploration — through one shared
  intent representation that unifies "what it sees" and "what it does."
- **Self-improving.** A guarded evolution loop reads its own codebase and goals
  and proposes improvements on a timer, behind immutable ethics guardrails it
  cannot rewrite.

---

## Architecture, at altitude

SignalBot wraps a foundation model in a scaffolding layer. The pieces (kept here
at the conceptual level — the implementations are the interesting part):

| System | What it does |
|---|---|
| **Memory layer** | Identity-modulated decay and consolidation — memories are scored by relevance, recency, and how well they fit *who the system is*, then pruned or archived accordingly. Not a vector-DB dump. |
| **Cognitive state vector** | A multi-dimensional internal state (curiosity, confidence, fatigue, …) that modulates both behavior and what gets remembered. |
| **Temporal daemon** | Continuous background cognitive cycles. The bot is "thinking" between your turns, not just when prompted. |
| **Goal / curiosity engine** | Forms and prioritizes its own goals, with good-sense gating that prevents runaway pursuit. Survival earns compute by merit, not by age. |
| **Initiative dispatcher** | When a goal clears a quality bar, the system can act on its own — including reaching out to you unprompted, under hard rate limits. |
| **Epistemic policy** | The `[GROUND]` / `[DREAM]` two-lane output discipline, threaded through every response. |
| **Ethics layer** | Immutable guardrails. The evolution loop and every autonomous process are forbidden from touching them. |
| **Embodiment bridge** | A shared intent representation that lets a text-only model "see" through a vision system and drive motors through the same channel — making perception and action model-agnostic too. |

How each of these actually works — the decay math, the intent encoding, the
consolidation strategy — lives in the source and the paper, and some of it
deliberately doesn't live here.

---

## What's new in v10

- **Initiative dispatcher + skill-spawn** — the autonomous-action executor: goals
  that clear the bar get dispatched (talk-initiative is live; agent-spawn lanes
  behind it), with rate limits and quiet-hours baked in, not bolted on.
- **Thought-quality ladder** — a merit-cycling goal engine. Ideas ascend tiers by
  earning it, paying compute proportional to the climb; junk dies cheaply at the
  bottom, worthy threads get adjudicated at the top.
- **Preference layer** — the system developing *its own* tastes (the "wanting" leg
  of the thesis): the slow integral of what it keeps coming back to, surviving
  restarts and model swaps.
- Memory-recall and parser hardening, cross-model "voice" evaluation, and a long
  tail of fixes from the development log.

---

## Status

Experimental, and under rapid (occasionally breaking) change — this is a working
research system, not a packaged product. Tuned for the author's hardware
(RTX 4060, Ubuntu); values may need adjustment for yours.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy requests        # plus torch/ultralytics for the vision + address models

python3 signalbot.py              # CLI overlay
```

Runs against the Anthropic API (Claude) or local models via llama.cpp / Ollama.
A web app and the robot platform have their own entry points.

---

## Background

There is a published paper — *Cognitive Overlay Architecture for Persistent LLM
Agency* (figshare) — and a pending Canadian patent (**CIPO Application No.
3,304,098**, filed March 2026).

Built solo, self-taught, entirely through chat interfaces and a terminal.

---

## The deep dive — by invitation

This repository is the showcase: the working overlay, enough to read the
architecture and run it. The full story lives a layer deeper and is shared with
collaborators, reviewers, and serious inquiries by invitation — the dated design
log (`DEVLOG`), the durable architecture notes, the autonomous self-evolution
loop, and the parts of the moat that aren't public for a reason.

If you've read this far and want to see under the hood, reach out below.

---

## License

**Dual-licensed:**

- **AGPL-3.0** — free for open-source, research, and personal use. Play with it,
  fork it, learn from it.
- **Commercial license** — required for proprietary or closed-source use. If
  you're building something that makes money on top of this, let's talk.

**Patent Pending** — Canadian Patent Application No. 3,304,098.

Commercial licensing inquiries: **crater_noggin@hotmail.com**
