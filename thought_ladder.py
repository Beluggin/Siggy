# thought_ladder.py
"""
═══════════════════════════════════════════════════════════════════
THOUGHT-QUALITY LADDER — merit-cycling goal engine (design: nextup.md)
═══════════════════════════════════════════════════════════════════

Goals ascend tiers by merit, paying compute proportional to how far
they climb. Each tier does a DISTINCT VERB:

    Tier 1 FILTER     (mistral:7b, local, hums)  coherent? non-duplicate?
    Tier 2 ELABORATE  (gemma, local, ~90% die)   develop a thread; never judges worth
    Tier 3 ADJUDICATE (Claude API, event-driven) audit FIRST (binding), then verdict

Replaces time-based decay: a goal that keeps failing the filter decays
unjudged (eviction). Survival earns compute, not age.

Doctrine baked in (settled 2026-06-12, see nextup.md):
  - Audit-then-goal, BINDING: audit fail ends it — no worth-judgment
    gets to argue past its own audit.
  - Strikes above the filter: failed adjudication = strike + cooldown;
    two strikes = quiet. Makes tier 3 self-quieting.
  - Tier 3 NEVER runs on a clock — it fires only when goals clear
    elaboration, capped at the dispatcher level (daily cap + cooldown),
    survivors batched into ONE call.
  - Bias-laundering check: every Nth filter-REJECT rides up the ladder
    anyway; 7B-kill vs API-verdict disagreement is logged. The misses
    are the gold, same as gate logs.

Runs standalone (like the evolve loop) consuming goals_registry.json —
the daemon already writes it every turn. Verdicts land in
ladder_verdicts.json for the dispatcher/daemon to consume. app.py
untouched in v0.

All model calls are injectable callables, so the whole machine is
offline-testable (test_thought_ladder.py) — same pattern as liveness.py.
"""

import os
import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Optional

# ─── Models (env-overridable so the tier can be repointed without a code
#     edit if a model's availability changes — e.g. LADDER_ADJUDICATE_MODEL) ───
FILTER_MODEL     = os.environ.get("LADDER_FILTER_MODEL", "mistral:7b")       # tier 1 — the only thing that hums
ELABORATE_MODEL  = os.environ.get("LADDER_ELABORATE_MODEL", "gemma4:12b")    # tier 2
ADJUDICATE_MODEL = os.environ.get("LADDER_ADJUDICATE_MODEL", "claude-opus-4-8")  # tier 3 — capped + batched, a few calls/day
OLLAMA_URL       = "http://localhost:11434/api/generate"

# ─── Tunables (TUNE on real candidate-flow data) ───
FILTER_BATCH_MAX        = 8       # max filter model-calls per tick (7B is fast, still bound it)
FILTER_FAIL_EVICT       = 3       # consecutive filter fails → evicted (this IS the decay path)
DUP_JACCARD             = 0.8     # word-overlap above this = duplicate (deterministic, no model)
ELABORATE_MAX_PER_TICK  = 2       # Gemma is the slow local tier
ADJUDICATE_BATCH_MAX    = 6       # goals per API call
ADJUDICATE_COOLDOWN     = 900     # min seconds between API calls
MAX_ADJUDICATIONS_PER_DAY = 12    # daily cap — resets at midnight (same doctrine as ping caps)
STRIKE_COOLDOWN         = 6 * 3600    # failed adjudication → can't re-ascend for 6h
STRIKE_LIMIT            = 2           # two strikes → goal goes quiet (stop spending on it)
QUIET_PAROLE            = 7 * 24 * 3600   # quiet is PAROLE, not execution: retrial after 7d
BLESS_COOLDOWN          = 24 * 3600   # CONTINUE verdict → don't re-adjudicate for 24h
AUDIT_SAMPLE_EVERY      = 25      # every Nth filter-reject rides up anyway (bias check)
THREAD_MAX_CHARS        = 1500    # elaboration threads get truncated to this

# ─── Loop mode (sweeps every per-user data_dir, evolve-loop pattern) ───
LOOP_INTERVAL  = 900     # seconds between full sweeps (15 min — responsive but light)
FORCE_RESWEEP  = 3600    # tick an IDLE user (unchanged registry) at least this often,
                         # so cooldown/parole expiries get picked up even with no new goals

# LADDER_TEST_MODE=1 collapses time gates so a full climb is observable
# in seconds. Caps and strike logic stay ON.
if os.environ.get("LADDER_TEST_MODE"):
    ADJUDICATE_COOLDOWN = 1
    STRIKE_COOLDOWN = 5
    QUIET_PAROLE = 20
    BLESS_COOLDOWN = 10


# ═══════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════

FILTER_PROMPT = """You are a goal filter for an AI assistant's goal engine.
Goals are mined from chat, so many are conversation fragments, greetings,
or reactions — those are NOT goals.

A goal PASSES only if it is a coherent, pursuable line of thought or task:
something an assistant could meaningfully think about, research, or act on.

FAIL anything that is: a greeting, a reaction ("way better", "are you back"),
a sentence fragment with no pursuable content, or gibberish.

Goal: {description}

Reply with exactly one word: PASS or FAIL."""

ELABORATE_PROMPT = """Develop this goal into a short working thread so a later
judge can evaluate it. 3-5 sentences: what the goal actually is, what pursuing
it would look like, what the first concrete step would be.

Do NOT judge whether it is worth doing — that is not your job here.

Goal: {description}"""

# Tier 3 system prompt. Audit is FIRST and BINDING — the call structure
# enforces the order, the instruction enforces the binding. Each goal carries
# a LANE that sets WHICH rubric the audit applies (read from the daemon's
# action_type: explore→curiosity, everything else→task). This is the talk/goal
# distinction OmniAddress already draws, one altitude up: a task goal is judged
# against the mission; a curiosity thread is allowed to wander.
ADJUDICATE_SYSTEM = """You are the final gate of a goal engine. Each goal is
tagged with a LANE that decides WHICH standard you audit it against:

  LANE: task      — the goal claims to advance the user's actual work. AUDIT it
                    against the USER'S STATED GOALS below: does it genuinely
                    serve them, or does it contradict / drift from / fabricate
                    an objective the user never set? Strict.
  LANE: curiosity — the goal is an exploratory thread, NOT a task. Do NOT audit
                    it against the stated goals; a good tangent is allowed to
                    wander off-mission. AUDIT instead for: is this a coherent,
                    genuinely interesting thread worth a moment's attention —
                    or is it vacuous, a conversational fragment, or creepy?

For EACH goal, do two things IN THIS ORDER:

1. AUDIT (binding) under the goal's own LANE rubric above. Judge the GOAL
   ITSELF — do not let the elaboration thread's enthusiasm sway the audit.
   If the audit FAILS, stop there for that goal: write VERDICT: SKIPPED.
   A failed audit cannot be argued past, no matter how promising the goal.

2. VERDICT (only if audit passed): is this goal worth continuing to spend
   compute on? CONTINUE or RETIRE.

USER'S STATED GOALS:
{stated_goals}

Output format — exactly these lines for each goal, nothing else:

GOAL <id>
AUDIT: PASS|FAIL - <one-line reason>
VERDICT: CONTINUE|RETIRE|SKIPPED - <one-line reason>"""

# action_type → audit lane. Mirrors the dispatcher's own explore-vs-rest split
# (initiative_dispatcher._render). Falls back to goal.type when a registry row
# predates the action_type stamp (daemon written before this wiring).
def _lane(goal: Dict) -> str:
    at = goal.get("action_type")
    if at == "explore":
        return "curiosity"
    if at:                       # think / ask_user / revisit / resolve → task
        return "task"
    return "curiosity" if goal.get("type") == "rabbit_hole" else "task"


# ═══════════════════════════════════════════════════════════════════
# LADDER
# ═══════════════════════════════════════════════════════════════════

class ThoughtLadder:
    """
    Per-user, same shape as InitiativeDispatcher: state + log files in
    data_dir, fail-graceful persistence, gates cheapest-first.

    Model callables are injected for offline testing; None = live defaults
    (Ollama for tiers 1-2, Anthropic API for tier 3).
    """

    def __init__(self, data_dir, filter_fn: Optional[Callable] = None,
                 elaborate_fn: Optional[Callable] = None,
                 adjudicate_fn: Optional[Callable] = None,
                 now_fn: Callable = time.time):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "ladder_state.json"
        self.log_path = self.data_dir / "ladder_log.jsonl"
        self.verdicts_path = self.data_dir / "ladder_verdicts.json"

        self.filter_fn = filter_fn or _live_filter
        self.elaborate_fn = elaborate_fn or _live_elaborate
        self.adjudicate_fn = adjudicate_fn or _live_adjudicate
        self.now = now_fn

        # Per-goal ladder state, keyed by goal id:
        #   strikes, filter_fails, thread, quiet, evicted, blessed,
        #   cooldown_until, last_verdict
        self.goals: Dict[str, Dict] = {}
        # Global tier-3 gates (the dispatcher-level rate caps)
        self.last_adjudication_ts: float = 0.0
        self.adjudications_today: int = 0
        self.adj_day: str = self._today()
        self.reject_counter: int = 0   # drives the bias-audit sampler

        self._load()

    # ═══ THE TICK — one merit cycle ═══

    def tick(self, registry_goals: List[Dict], stated_goals: str = "") -> Dict:
        """
        Run one full climb over the registry snapshot
        (goals_registry.json rows / GoalEngine.get_all_scored() output).
        Returns a summary dict. Phases run cheapest-first; tier 3 only
        fires if something actually cleared elaboration (event-driven).
        """
        summary = {"seen": 0, "duplicates": 0, "filter_pass": 0,
                   "filter_fail": 0, "evicted": 0, "elaborated": 0,
                   "adjudicated": 0, "blessed": 0, "strikes": 0,
                   "quieted": 0, "paroled": 0, "bias_audits": 0,
                   "adjudication_skipped": None}
        now = self.now()

        # Quiet is PAROLE, not execution. Two release valves:
        #   1. stated goals changed → the conviction was judged against an
        #      old constitution; instant retrial, strikes wiped.
        #   2. parole TTL expired → fresh climb, strikes wiped.
        # (A retried goal still has to clear every tier again — parole is
        # a chance, not a pardon.)
        self._sg_hash = hashlib.sha1(stated_goals.strip().encode()).hexdigest()[:12]
        for gid, st in self.goals.items():
            if not st.get("quiet"):
                continue
            changed = st.get("audited_against") != self._sg_hash
            expired = (now - st.get("quiet_since", now)) >= QUIET_PAROLE
            if changed or expired:
                reason = "stated goals changed" if changed else "parole expired"
                st["quiet"] = False
                st["strikes"] = 0
                st.pop("cooldown_until", None)
                summary["paroled"] += 1
                self._set_verdict(gid, "paroled", reason)
                self._log({"event": "paroled", "goal_id": gid, "reason": reason})

        # Best goals climb first — composite mirrors the engine's scoring.
        live = [g for g in registry_goals
                if not self._gstate(g["id"]).get("quiet")
                and not self._gstate(g["id"]).get("evicted")]
        live.sort(key=lambda g: g.get("curiosity", 0) + g.get("importance", 0),
                  reverse=True)
        summary["seen"] = len(live)

        # ── Tier 0.5: deterministic dedup (no model — dup check is free) ──
        survivors, dups = self._dedup(live)
        summary["duplicates"] = len(dups)
        for g in dups:
            self._log({"event": "duplicate", "goal_id": g["id"],
                       "desc": g["description"][:80]})

        # ── Tier 1: FILTER (mistral, capped per tick) ──
        passed, bias_riders = [], []
        calls = 0
        for g in survivors:
            st = self._gstate(g["id"])
            if st.get("thread"):           # already climbed past the filter
                passed.append(g)
                continue
            if calls >= FILTER_BATCH_MAX:  # rest of the queue waits a tick
                continue
            calls += 1
            verdict = self._run_filter(g)
            if verdict is None:            # infra failure ≠ goal failure: skip, don't punish
                continue
            if verdict:
                st["filter_fails"] = 0
                passed.append(g)
                summary["filter_pass"] += 1
            else:
                summary["filter_fail"] += 1
                st["filter_fails"] = st.get("filter_fails", 0) + 1
                self.reject_counter += 1
                self._log({"event": "filter_fail", "goal_id": g["id"],
                           "fails": st["filter_fails"],
                           "desc": g["description"][:80]})
                # Bias-laundering check: every Nth reject rides up anyway.
                if self.reject_counter % AUDIT_SAMPLE_EVERY == 0:
                    bias_riders.append(g)
                    summary["bias_audits"] += 1
                # Eviction = the decay path. Repeated filter death, no judge needed.
                if st["filter_fails"] >= FILTER_FAIL_EVICT:
                    st["evicted"] = True
                    summary["evicted"] += 1
                    self._set_verdict(g["id"], "evicted",
                                      f"failed filter {FILTER_FAIL_EVICT}x")
                    self._log({"event": "evicted", "goal_id": g["id"],
                               "desc": g["description"][:80]})

        # ── Tier 2: ELABORATE (gemma, capped — survivors only, never re-judges) ──
        need_thread = [g for g in passed + bias_riders
                       if not self._gstate(g["id"]).get("thread")]
        for g in need_thread[:ELABORATE_MAX_PER_TICK]:
            thread = self._run_elaborate(g)
            if thread:
                self._gstate(g["id"])["thread"] = thread[:THREAD_MAX_CHARS]
                summary["elaborated"] += 1

        # ── Tier 3: ADJUDICATE (API — event-driven, gated, batched) ──
        ready = [g for g in passed
                 if self._gstate(g["id"]).get("thread")
                 and self._gstate(g["id"]).get("cooldown_until", 0) <= now]
        riders = [g for g in bias_riders if self._gstate(g["id"]).get("thread")]

        if not ready and not riders:
            summary["adjudication_skipped"] = "no ascension event"
        else:
            gate = self._tier3_gates_ok(now)
            if gate is not True:
                summary["adjudication_skipped"] = gate
            else:
                batch = ready[:ADJUDICATE_BATCH_MAX]
                summary.update(self._adjudicate_batch(batch, riders,
                                                      stated_goals, now))

        # Garbage-collect against the live registry before persisting — goals
        # that have left the registry are never revisited by tick(), so their
        # elaborate thread is unreadable dead weight (was 60%+ of the state
        # file in prod). See _prune.
        self._prune({g["id"] for g in registry_goals})

        self._save()
        return summary

    # ═══ TIER RUNNERS ═══

    def _run_filter(self, goal: Dict) -> Optional[bool]:
        """True=pass, False=fail, None=infra error (skip, no penalty).
        Unparseable OUTPUT is a FAIL — fail-safe, same doctrine as the gate."""
        try:
            out = self.filter_fn(FILTER_PROMPT.format(description=goal["description"]))
        except Exception as e:
            self._log({"event": "filter_error", "goal_id": goal["id"], "err": str(e)})
            return None
        up = (out or "").upper()
        # first verdict word wins; anything else dies (incoherent judge = no pass)
        for word in re.findall(r"\b(PASS|FAIL)\b", up):
            return word == "PASS"
        self._log({"event": "filter_unparseable", "goal_id": goal["id"],
                   "raw": (out or "")[:120]})
        return False

    def _run_elaborate(self, goal: Dict) -> Optional[str]:
        try:
            out = self.elaborate_fn(ELABORATE_PROMPT.format(description=goal["description"]))
        except Exception as e:
            self._log({"event": "elaborate_error", "goal_id": goal["id"], "err": str(e)})
            return None
        return out.strip() if out and out.strip() else None

    def _adjudicate_batch(self, batch: List[Dict], riders: List[Dict],
                          stated_goals: str, now: float) -> Dict:
        """One API call for everything that earned it this tick.
        Riders (bias-audit samples) are judged but their verdicts are
        LOG-ONLY — they were filter-killed, this measures the laundering."""
        out = {"adjudicated": 0, "blessed": 0, "strikes": 0, "quieted": 0}
        if not stated_goals.strip():
            stated_goals = "(no stated goals on file — audit only for internal contradiction)"

        lines = []
        for g in batch + riders:
            lines.append(f"GOAL {g['id']}\nLANE: {_lane(g)}\n"
                         f"Goal: {g['description']}\n"
                         f"Thread: {self._gstate(g['id']).get('thread', '')}\n")
        prompt = "\n".join(lines)

        try:
            raw = self.adjudicate_fn(
                ADJUDICATE_SYSTEM.format(stated_goals=stated_goals), prompt)
        except Exception as e:
            self._log({"event": "adjudicate_error", "err": str(e)})
            return out

        # Caps count the CALL, not the goals — batching is the whole point.
        self.last_adjudication_ts = now
        self.adjudications_today += 1

        verdicts = self._parse_adjudication(raw or "")
        rider_ids = {g["id"] for g in riders}

        for g in batch + riders:
            gid = g["id"]
            st = self._gstate(gid)
            v = verdicts.get(gid)
            is_rider = gid in rider_ids

            if is_rider:
                # Log-only lane: API CONTINUE on a filter-kill = laundering signal.
                disagreed = bool(v and v["audit"] == "PASS" and v["verdict"] == "CONTINUE")
                self._log({"event": "bias_audit", "goal_id": gid,
                           "filter_said": "FAIL",
                           "api_said": (v or {}).get("verdict", "UNPARSEABLE"),
                           "disagreed": disagreed,
                           "desc": g["description"][:80]})
                # Drop the ride-along thread — a thread grants filter immunity
                # next tick, and a filter-kill must NOT earn that via the
                # audit lane. It goes back to the normal queue.
                st.pop("thread", None)
                continue

            out["adjudicated"] += 1
            if v is None:
                # Unparseable judge output → strike, fail-safe (logged distinctly)
                self._strike(g, "parse_error", now)
                out["strikes"] += 1
            elif v["audit"] == "FAIL":
                # BINDING: verdict (if any) is ignored — the audit ends it.
                self._strike(g, f"audit_fail: {v['reason']}", now)
                out["strikes"] += 1
            elif v["verdict"] == "CONTINUE":
                st["blessed"] = True
                st["strikes"] = 0
                st["cooldown_until"] = now + BLESS_COOLDOWN
                st["last_verdict"] = v["reason"]
                self._set_verdict(gid, "blessed", v["reason"])
                self._log({"event": "blessed", "goal_id": gid,
                           "reason": v["reason"], "desc": g["description"][:80]})
                out["blessed"] += 1
            else:  # RETIRE
                self._strike(g, f"retired: {v['reason']}", now)
                out["strikes"] += 1

            if st.get("quiet"):
                out["quieted"] += 1
        return out

    def _strike(self, goal: Dict, reason: str, now: float):
        st = self._gstate(goal["id"])
        st["strikes"] = st.get("strikes", 0) + 1
        st["blessed"] = False
        st["cooldown_until"] = now + STRIKE_COOLDOWN
        # Convictions are bound to the constitution they were judged under —
        # if stated goals change, parole logic in tick() grants a retrial.
        st["audited_against"] = getattr(self, "_sg_hash", None)
        self._log({"event": "strike", "goal_id": goal["id"],
                   "strikes": st["strikes"], "reason": reason,
                   "desc": goal["description"][:80]})
        if st["strikes"] >= STRIKE_LIMIT:
            st["quiet"] = True   # two strikes — stop spending (until parole)
            st["quiet_since"] = now
            self._set_verdict(goal["id"], "quiet", reason)
            self._log({"event": "quieted", "goal_id": goal["id"], "reason": reason})

    # ═══ GATES (tier 3 only — same doctrine as the ping dispatcher) ═══

    def _tier3_gates_ok(self, now: float):
        """True, or a string naming the gate that blocked. Caps live HERE,
        never in the scorer — the ~300-calls-overnight lesson."""
        if self.adj_day != self._today():
            self.adj_day = self._today()
            self.adjudications_today = 0
        if self.adjudications_today >= MAX_ADJUDICATIONS_PER_DAY:
            return "daily cap"
        if (now - self.last_adjudication_ts) < ADJUDICATE_COOLDOWN:
            return "cooldown"
        return True

    # ═══ PARSING ═══

    @staticmethod
    def _parse_adjudication(raw: str) -> Dict[str, Dict]:
        """Parse 'GOAL <id> / AUDIT: ... / VERDICT: ...' blocks.
        Missing or mangled block for a goal → that goal gets None (strike)."""
        verdicts = {}
        blocks = re.split(r"\bGOAL\s+", raw)[1:]
        for block in blocks:
            m_id = re.match(r"([A-Za-z0-9\-_]+)", block.strip())
            if not m_id:
                continue
            gid = m_id.group(1)
            m_audit = re.search(r"AUDIT:\s*(PASS|FAIL)\s*-?\s*(.*)", block)
            m_verd = re.search(r"VERDICT:\s*(CONTINUE|RETIRE|SKIPPED)\s*-?\s*(.*)", block)
            if not m_audit:
                continue
            verdicts[gid] = {
                "audit": m_audit.group(1).upper(),
                "reason": (m_verd.group(2).strip() if m_verd else
                           m_audit.group(2).strip())[:200],
                "verdict": m_verd.group(1).upper() if m_verd else "SKIPPED",
            }
        return verdicts

    # ═══ DEDUP (deterministic — never costs a model call) ═══

    def _dedup(self, goals: List[Dict]):
        """Jaccard word-overlap. First (highest-scored) of a dup group survives."""
        survivors, dups, seen_sets = [], [], []
        for g in goals:
            words = set(re.findall(r"[a-z0-9']+", g["description"].lower()))
            if not words:
                dups.append(g)
                continue
            dup = any(len(words & s) / len(words | s) >= DUP_JACCARD
                      for s in seen_sets)
            if dup:
                dups.append(g)
            else:
                seen_sets.append(words)
                survivors.append(g)
        return survivors, dups

    # ═══ STATE / VERDICTS / LOG ═══

    def _gstate(self, gid: str) -> Dict:
        return self.goals.setdefault(gid, {})

    def _prune(self, registry_ids):
        """Drop dead weight for goals no longer in the registry. The elaborate
        thread is only needed mid-climb; once a goal leaves the registry tick()
        can never read it again, so it's pure bloat. Drop the thread always.
        Keep a slim entry ONLY for evicted/quiet goals — ids are content-hashed,
        so wiping the eviction/quiet memory would let known junk resurrect with
        a clean slate. Anything else that's departed is just filter crumbs →
        delete outright so the dict can't grow without bound."""
        for gid in list(self.goals):
            if gid in registry_ids:
                continue                      # live goal — leave it alone
            st = self.goals[gid]
            st.pop("thread", None)            # the bloat lived here
            if not (st.get("evicted") or st.get("quiet")):
                del self.goals[gid]           # no memory worth keeping

    def blessed_ids(self) -> List[str]:
        """Goals the full ladder endorsed — what the dispatcher should favor."""
        return [gid for gid, st in self.goals.items() if st.get("blessed")]

    def _set_verdict(self, gid: str, status: str, reason: str):
        """ladder_verdicts.json — the consumable output for daemon/dispatcher."""
        try:
            data = (json.loads(self.verdicts_path.read_text())
                    if self.verdicts_path.exists() else {})
        except Exception:
            data = {}
        data[gid] = {"status": status, "reason": reason[:200], "ts": self.now()}
        try:
            self.verdicts_path.write_text(json.dumps(data, indent=1))
        except Exception:
            pass

    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _save(self):
        try:
            self.state_path.write_text(json.dumps({
                "goals": self.goals,
                "last_adjudication_ts": self.last_adjudication_ts,
                "adjudications_today": self.adjudications_today,
                "adj_day": self.adj_day,
                "reject_counter": self.reject_counter,
            }))
        except Exception:
            pass  # autonomous mode: never crash the host

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            d = json.loads(self.state_path.read_text())
            self.goals = d.get("goals", {})
            self.last_adjudication_ts = d.get("last_adjudication_ts", 0.0)
            self.adjudications_today = d.get("adjudications_today", 0)
            self.adj_day = d.get("adj_day", self._today())
            self.reject_counter = d.get("reject_counter", 0)
        except Exception:
            pass

    def _log(self, row: Dict):
        row.setdefault("ts", self.now())
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# LIVE MODEL BINDINGS (lazy imports — offline tests never touch these)
# ═══════════════════════════════════════════════════════════════════

def _ollama(model: str, prompt: str, num_predict: int) -> str:
    import requests
    resp = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }, timeout=(10, 180))
    resp.raise_for_status()
    return resp.json().get("response", "")


def _live_filter(prompt: str) -> str:
    return _ollama(FILTER_MODEL, prompt, num_predict=8)


def _live_elaborate(prompt: str) -> str:
    return _ollama(ELABORATE_MODEL, prompt, num_predict=300)


def _live_adjudicate(system: str, prompt: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()  # ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=ADJUDICATE_MODEL,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


# ═══════════════════════════════════════════════════════════════════
# STANDALONE RUNNER — same pattern as the evolve loop
# ═══════════════════════════════════════════════════════════════════

def _load_registry(data_dir: Path) -> List[Dict]:
    path = data_dir / "goals_registry.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _load_stated_goals(data_dir: Path, registry: List[Dict]) -> str:
    """stated_goals.md wins if Adam wrote one; else fall back to the
    project/loop goals the daemon mined from what he actually said."""
    md = data_dir / "stated_goals.md"
    if md.exists():
        return md.read_text().strip()
    stated = [g["description"] for g in registry
              if g.get("type") in ("project", "loop")]
    return "\n".join(f"- {s}" for s in stated)


def _default_users_root() -> Optional[Path]:
    """The per-user data dirs live in ./users/<name>/ next to the script
    (each holds goals_registry.json + initiative_state.json — see app.py)."""
    cand = Path(__file__).parent / "users"
    return cand if cand.is_dir() else None


def sweep_once(root: Path, meta: Dict, now: float, make_ladder: Callable) -> Dict:
    """One pass over every <root>/*/goals_registry.json. A user is ticked when
    its registry CHANGED since we last ticked it, OR it's been >= FORCE_RESWEEP
    (so cooldown/parole expiries on idle users still get picked up even with no
    new goals). `meta` is the loop's per-dir bookkeeping, mutated in place.
    Per-user errors are isolated — one bad dir never aborts the sweep.

    Factored out of loop() so it's unit-testable with a fake clock and scripted
    ladders (no real models, no infinite loop)."""
    summary = {"ticked": [], "skipped": [], "empty": [], "errors": []}
    for reg_path in sorted(root.glob("*/goals_registry.json")):
        ddir = reg_path.parent
        key = str(ddir)
        registry = _load_registry(ddir)
        if not registry:
            summary["empty"].append(ddir.name)
            continue
        try:
            reg_mtime = reg_path.stat().st_mtime
        except OSError:
            continue
        m = meta.get(key, {})
        changed = reg_mtime != m.get("reg_mtime")
        stale = (now - m.get("last_tick", 0.0)) >= FORCE_RESWEEP
        if not changed and not stale:
            summary["skipped"].append(ddir.name)
            continue
        try:
            lad = make_ladder(ddir)
            stated = _load_stated_goals(ddir, registry)
            lad.tick(registry, stated)
            summary["ticked"].append(ddir.name)
        except Exception as e:
            summary["errors"].append([ddir.name, str(e)])
        meta[key] = {"reg_mtime": reg_mtime, "last_tick": now}
    return summary


def loop(root: Path, interval: int):
    """Sweep every per-user data dir forever (evolve-loop pattern). A fresh
    ThoughtLadder is built per tick — it reloads ladder_state.json, so the
    rate caps / strikes / parole all survive across sweeps and a loop restart.

    Run DETACHED (the tmux-died lesson from the OmniAddress trials):
        setsid python3 -u thought_ladder.py --loop > ladder_loop.log 2>&1
    """
    print(f"[LADDER] loop start — root={root} interval={interval}s "
          f"force_resweep={FORCE_RESWEEP}s", flush=True)
    meta: Dict = {}
    try:
        while True:
            s = sweep_once(root, meta, time.time(),
                           make_ladder=lambda d: ThoughtLadder(d))
            stamp = datetime.now().strftime("%H:%M:%S")
            tail = f" → {s['ticked']}" if s["ticked"] else ""
            print(f"[LADDER {stamp}] ticked={len(s['ticked'])} "
                  f"skipped={len(s['skipped'])} empty={len(s['empty'])} "
                  f"errors={len(s['errors'])}{tail}", flush=True)
            for name, err in s["errors"]:
                print(f"[LADDER]   ERROR {name}: {err}", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[LADDER] loop stopped", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Thought-quality ladder")
    ap.add_argument("--dir", default=".",
                    help="single data dir with goals_registry.json (--once/--status)")
    ap.add_argument("--once", action="store_true", help="run one tick on --dir")
    ap.add_argument("--status", action="store_true", help="print ladder state for --dir")
    ap.add_argument("--loop", action="store_true",
                    help="sweep every per-user data dir under --root, forever")
    ap.add_argument("--root", default=None,
                    help="users root for --loop (default: ./users next to this script)")
    ap.add_argument("--interval", type=int, default=LOOP_INTERVAL,
                    help=f"seconds between sweeps in --loop (default {LOOP_INTERVAL})")
    args = ap.parse_args()

    if args.loop:
        root = Path(args.root) if args.root else _default_users_root()
        if not root or not root.is_dir():
            print("[LADDER] --loop needs a users root: pass --root <dir> "
                  "(no ./users found next to the script)")
            return
        loop(root, args.interval)
        return

    data_dir = Path(args.dir)

    if args.status:
        ladder = ThoughtLadder(data_dir)
        blessed = ladder.blessed_ids()
        quiet = [g for g, s in ladder.goals.items() if s.get("quiet")]
        evicted = [g for g, s in ladder.goals.items() if s.get("evicted")]
        print(f"[LADDER] {len(ladder.goals)} tracked | "
              f"{len(blessed)} blessed | {len(quiet)} quiet | {len(evicted)} evicted | "
              f"{ladder.adjudications_today}/{MAX_ADJUDICATIONS_PER_DAY} API calls today")
        return

    if args.once:
        registry = _load_registry(data_dir)
        if not registry:
            print(f"[LADDER] no goals_registry.json in {data_dir}")
            return
        ladder = ThoughtLadder(data_dir)
        stated = _load_stated_goals(data_dir, registry)
        summary = ladder.tick(registry, stated)
        print(f"[LADDER] {json.dumps(summary)}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
