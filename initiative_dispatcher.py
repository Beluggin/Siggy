# initiative_dispatcher.py
"""
═══════════════════════════════════════════════════════════════════
INITIATIVE DISPATCHER — Lane 1: talk-initiative
═══════════════════════════════════════════════════════════════════

The missing executor for the daemon's ActionCandidate queue.
The daemon scores and prioritizes candidates every cycle (phases 2-4),
but until now nothing dispatched on them — a motor cortex with no
spinal cord. This is the spinal cord, lane 1 only:

    top candidate clears all guardrails → daemon-initiated chat message

GUARDRAIL DOCTRINE (the ~300-API-calls-overnight incident is the
backfire mode — rate limiting lives HERE, at the dispatcher, never
in the scorer):
  - score threshold      (candidate must clear it)
  - cooldown             (min seconds between pings)
  - daily cap            (max pings per day, resets at midnight)
  - quiet hours          (no pings overnight)
  - idle requirement     (never mid-conversation)
  - liveness leash       (pinging is atelic durative behavior — an
                          unanswered ping settles `false`; ignored
                          twice in a row → backoff doubles and the
                          goal goes quiet)

v0 costs ZERO API calls: the message text comes from
GoalEngine.generate_goal_prompt() (deterministic templates).
Upgrading to LLM-rendered pings later only changes _render().

Every fire and every settle is logged to initiative_log.jsonl —
ignored-vs-answered is free training signal on ping-worthiness
(the misses are the gold, same as gate logs).

THOUGHT-LADDER INTEGRATION (2026-06-13): candidate selection now reads
thought_ladder's verdicts (ladder_verdicts.json). quiet/evicted goals are
skipped (kills junk pings the daemon's intensity-only score would otherwise
fire); BLESSED goals get a composite bonus (quality over intensity). Soft
bias, not a hard gate — unverdicted goals still ping, so initiative doesn't
starve while the slow ladder catches up. The lane the ladder judged on is the
same talk/goal split _render already uses (explore→curiosity prompt).
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# ─── Tunables (TUNE on real candidate-flow data — that's what lane 1
#     exists to generate before lane 2's threshold gets set) ───
SCORE_THRESHOLD   = 0.75    # candidate composite_score must clear this
COOLDOWN_SECONDS  = 1800    # 30 min between pings (web-scale, not CLI's 45s)
MAX_PINGS_PER_DAY = 6
QUIET_START_HOUR  = 22      # no pings 22:00 → 08:00 local
QUIET_END_HOUR    = 8
IDLE_SECONDS      = 600     # user must be quiet 10 min — never mid-conversation
ANSWER_WINDOW     = 1800    # user turn within 30 min of a ping = "answered"
IGNORE_LIMIT      = 2       # ignored twice in a row → backoff + quiet the goal
LADDER_BLESSED_BONUS = 0.30 # thought_ladder verdict integration: a goal the
                            # ladder BLESSED (cleared its binding audit) gets
                            # this added to composite_score — effectively lowers
                            # its ping threshold, so a high-QUALITY goal can ping
                            # at moderate intensity. quiet/evicted ladder goals
                            # are skipped outright. Soft bias, NOT a hard gate:
                            # unverdicted goals still ping on raw score, so
                            # initiative doesn't starve while the (slow) ladder
                            # catches up. Verdicts read from ladder_verdicts.json.

# INITIATIVE_TEST_MODE=1 collapses the time gates so a fire is observable
# in seconds instead of tens of minutes. Caps and threshold stay ON.
if os.environ.get("INITIATIVE_TEST_MODE"):
    COOLDOWN_SECONDS = 10
    IDLE_SECONDS = 5
    ANSWER_WINDOW = 60


DEFAULT_IDENTITY = "You are SignalBot. Clever, candid, and slightly irreverent."


def build_initiative_prompt(identity: str, thread: str, action: str = "think",
                            reasoning: str = "", recent: str = "") -> str:
    """The talk-lane prompt for an initiative ping. PURE string building — no
    model, no Flask, no I/O — so app.py's live renderer and the model-swap test
    (initiative_swap_test.py) fire the IDENTICAL prompt. If this and the live
    path ever diverge the swap experiment stops being valid, so they share this.

    action == "explore" → curiosity tangent (allowed to wander); anything else
    → task-shaped (push it forward). Mirrors the thought-ladder lane split."""
    if action == "explore":
        aim = ("This is a curiosity thread of your OWN — a tangent you find "
               "genuinely interesting. Share the actual idea or the angle that "
               "hooks you and invite them in.")
    else:
        aim = ("Move this FORWARD: propose a concrete next step, a fresh angle, "
               "or ask one sharp question that advances it.")

    return "\n".join([
        identity or DEFAULT_IDENTITY,
        "",
        "### YOUR OWN INITIATIVE ###",
        "The user is away. Between messages you've kept turning something over "
        "on your own, and you've decided to reach out first. The thread you've "
        "been mulling (this is a rough note you captured from an earlier "
        "conversation, NOT something to read back to them):",
        f'  "{thread}"',
        (f"  (why it's on your mind: {reasoning})" if reasoning else ""),
        "",
        "### RECENT CONVERSATION ###",
        recent if recent else "(nothing recent)",
        "",
        "### WRITE THE MESSAGE ###",
        aim,
        "Hard rules:",
        "- Do NOT quote, paraphrase, or repeat their words back at them. They "
        "know what they said; echoing it reads as a broken bot.",
        "- Speak as yourself, your voice — a thought YOU had, not a prompt.",
        "- 1–3 sentences, natural, like texting someone you think about. No "
        "preamble like \"I was just thinking\" every time.",
        "- It's speculative/your own initiative, so open with [DREAM].",
        "",
        "Your message:",
    ])


class InitiativeDispatcher:
    """
    Per-user, lives on UserCognition next to the daemon.
    poll() is called by the web frontend's timer; notice_user_turn()
    is called from /api/chat so the dispatcher knows when the user
    is actually talking.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        # State persistence is what makes the rate caps real — if this
        # dir is missing, _save() would silently no-op and a restart
        # would reset every cap. Make sure it exists.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "initiative_state.json"
        self.log_path = self.data_dir / "initiative_log.jsonl"

        # Defaults — overwritten by _load() if state file exists.
        # last_user_turn_ts starts at boot time so a fresh server
        # still has to wait out the idle window before pinging.
        self.last_fired_ts: float = 0.0
        self.last_user_turn_ts: float = time.time()
        self.fired_today: int = 0
        self.fired_day: str = self._today()
        self.ignore_streak: int = 0
        self.backoff_mult: float = 1.0          # doubles on leash trip
        self.quiet_goals: set = set()           # goal_ids that ignored us twice
        self.pending: Optional[Dict] = None     # last fired, not yet settled

        # thought_ladder verdicts (ladder_verdicts.json), mtime-cached so poll()
        # doesn't hit disk every tick. Empty until the ladder has run → the
        # dispatcher behaves exactly as it did before the ladder existed.
        self.verdicts_path = self.data_dir / "ladder_verdicts.json"
        self._verdicts_cache: Dict = {}
        self._verdicts_mtime: float = -1.0

        self._load()

    # ═══ PUBLIC INTERFACE ═══

    def poll(self, daemon, goal_engine, render_fn=None) -> Optional[Dict[str, Any]]:
        """
        Called on the frontend's timer. Returns a ping dict
        {message, goal_id, action_type, score} or None.
        Order matters: settle first, then gates cheapest-first.

        render_fn (supplied by app.py in production) is the talk-lane LLM
        renderer: candidate dict → a genuine forward-moving message in the
        bot's own voice. When it's None (offline tests, no model) the render
        falls back to the deterministic goal-engine template.
        """
        now = time.time()
        self._settle_pending(now)   # an old unanswered ping settles false here

        if not self._gates_ok(now):
            return None

        candidate = self._pick_candidate(daemon)
        if candidate is None:
            return None

        message = self._render(candidate, goal_engine, render_fn)
        if not message:
            return None

        # Fire.
        self.last_fired_ts = now
        self.fired_today += 1
        self.pending = {
            "ts": now,
            "goal_id": candidate["goal_id"],
            "action_type": candidate["action_type"],
            "score": candidate["composite_score"],
            # what the ladder thought at fire time — joins to the answered/
            # ignored leash events (same goal_id) so blessed-but-ignored vs
            # unblessed-but-answered becomes a measurable signal on whether the
            # ladder's "worth it" matches real engagement (misses are gold).
            "ladder_status": self._ladder_verdicts().get(
                candidate["goal_id"], {}).get("status"),
            "message": message,
        }
        self._log({"event": "fired", **self.pending})
        self._save()
        return dict(self.pending)

    def notice_user_turn(self):
        """
        Called from /api/chat on every real user message.
        A user turn while a ping is pending = the ping was answered
        (settles done): leash resets, backoff clears.
        """
        now = time.time()
        self.last_user_turn_ts = now
        if self.pending is not None:
            self._log({"event": "answered", "ts": now,
                       "goal_id": self.pending["goal_id"],
                       "latency": now - self.pending["ts"]})
            self.pending = None
            self.ignore_streak = 0
            self.backoff_mult = 1.0
        self._save()

    # ═══ GATES ═══

    def _gates_ok(self, now: float) -> bool:
        # quiet hours — local time
        hour = datetime.now().hour
        if QUIET_START_HOUR > QUIET_END_HOUR:   # window wraps midnight
            if hour >= QUIET_START_HOUR or hour < QUIET_END_HOUR:
                return False
        elif QUIET_START_HOUR <= hour < QUIET_END_HOUR:
            return False

        # daily cap — resets when the date changes
        if self.fired_day != self._today():
            self.fired_day = self._today()
            self.fired_today = 0
        if self.fired_today >= MAX_PINGS_PER_DAY:
            return False

        # cooldown × leash backoff
        if (now - self.last_fired_ts) < COOLDOWN_SECONDS * self.backoff_mult:
            return False

        # never mid-conversation
        if (now - self.last_user_turn_ts) < IDLE_SECONDS:
            return False

        # one ping in flight at a time
        if self.pending is not None:
            return False

        return True

    def _settle_pending(self, now: float):
        """Unanswered past the window → settles FALSE. Twice in a row →
        backoff doubles and that goal goes quiet (stop nagging)."""
        if self.pending is None:
            return
        if (now - self.pending["ts"]) < ANSWER_WINDOW:
            return
        self.ignore_streak += 1
        gid = self.pending["goal_id"]
        self._log({"event": "ignored", "ts": now, "goal_id": gid,
                   "ignore_streak": self.ignore_streak})
        if self.ignore_streak >= IGNORE_LIMIT:
            self.backoff_mult = min(8.0, self.backoff_mult * 2)
            self.quiet_goals.add(gid)
            self._log({"event": "leash", "ts": now, "goal_id": gid,
                       "backoff_mult": self.backoff_mult})
        self.pending = None
        self._save()

    # ═══ CANDIDATE SELECTION ═══

    def _pick_candidate(self, daemon) -> Optional[Dict]:
        """Top recommendation that survives BOTH quality gates:
          - leash quiet-set (ignored twice) OR ladder quiet/evicted → skipped
          - ladder BLESSED → composite gets LADDER_BLESSED_BONUS, floating a
            high-quality goal up the queue and letting it ping at moderate
            intensity (quality over intensity).
        Soft bias: unverdicted goals still compete on raw composite, so the
        dispatcher never goes mute waiting on the slow ladder."""
        try:
            snap = daemon.get_snapshot()
        except Exception:
            return None
        verdicts = self._ladder_verdicts()
        ranked = []
        for rec in snap.top_recommendations:
            gid = rec.get("goal_id")
            status = verdicts.get(gid, {}).get("status")
            # hard skip: user ignored it twice, OR the ladder said don't bother
            if gid in self.quiet_goals or status in ("quiet", "evicted"):
                continue
            eff = rec.get("composite_score", 0.0)
            if status == "blessed":
                eff += LADDER_BLESSED_BONUS
            ranked.append((eff, rec))
        # blessed goals float up; equal scores keep the daemon's order (stable
        # sort, and we never compare the dicts — key is the float alone)
        ranked.sort(key=lambda er: er[0], reverse=True)
        for eff, rec in ranked:
            if eff >= SCORE_THRESHOLD:
                return rec
        return None

    def _ladder_verdicts(self) -> Dict:
        """Read ladder_verdicts.json (written by thought_ladder), mtime-cached.
        Missing file or parse error → last good cache (or {}), so a ladder that
        hasn't run yet is a silent no-op, not a failure."""
        try:
            mtime = self.verdicts_path.stat().st_mtime
        except OSError:
            return self._verdicts_cache
        if mtime != self._verdicts_mtime:
            try:
                self._verdicts_cache = json.loads(self.verdicts_path.read_text())
                self._verdicts_mtime = mtime
            except Exception:
                pass  # keep last good cache on a transient read/parse failure
        return self._verdicts_cache

    def _render(self, candidate: Dict, goal_engine, render_fn=None) -> Optional[str]:
        """Turn the chosen candidate into the ping TEXT.

        PRODUCTION (render_fn supplied): an LLM talk-lane call generates a
        genuine forward-moving thought — a next step, a fresh angle, a sharp
        question — in the bot's own voice, grounded in the goal thread +
        recent conversation, explicitly told NOT to quote the user back. This
        is the fix for the parroting: the daemon's goal description is often a
        raw captured user utterance, and the old template just wrapped it
        verbatim ("I've been chewing on: <your exact words>"). If render_fn
        fails or returns nothing we SKIP this tick (return None) rather than
        fall back to the echo template — a missed ping beats a parroted one.

        OFFLINE (render_fn is None — tests, no model): the deterministic
        goal-engine template. Preserves the original zero-API behaviour."""
        if render_fn is not None:
            try:
                msg = render_fn(candidate)
            except Exception:
                return None   # transient LLM/infra failure → no ping, not an echo
            return msg.strip() if (msg and msg.strip()) else None

        gid = candidate["goal_id"]
        if candidate.get("action_type") == "explore":
            return goal_engine.generate_curiosity_prompt(gid)
        return goal_engine.generate_goal_prompt(gid)

    # ═══ PERSISTENCE (survives server restarts so caps can't be
    #     reset by bouncing the process) ═══

    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _save(self):
        try:
            self.state_path.write_text(json.dumps({
                "last_fired_ts": self.last_fired_ts,
                "last_user_turn_ts": self.last_user_turn_ts,
                "fired_today": self.fired_today,
                "fired_day": self.fired_day,
                "ignore_streak": self.ignore_streak,
                "backoff_mult": self.backoff_mult,
                "quiet_goals": list(self.quiet_goals),
                "pending": self.pending,
            }))
        except Exception:
            pass  # autonomous mode: fail gracefully, never crash the app

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            d = json.loads(self.state_path.read_text())
            self.last_fired_ts = d.get("last_fired_ts", 0.0)
            self.last_user_turn_ts = d.get("last_user_turn_ts", time.time())
            self.fired_today = d.get("fired_today", 0)
            self.fired_day = d.get("fired_day", self._today())
            self.ignore_streak = d.get("ignore_streak", 0)
            self.backoff_mult = d.get("backoff_mult", 1.0)
            self.quiet_goals = set(d.get("quiet_goals", []))
            self.pending = d.get("pending")
        except Exception:
            pass

    def _log(self, row: Dict):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass
