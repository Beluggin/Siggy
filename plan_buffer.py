# plan_buffer.py
"""
═══════════════════════════════════════════════════════════════════
PLAN BUFFER — The Babysitter / Gatekeeper / Commitment Layer
═══════════════════════════════════════════════════════════════════

PURPOSE:
  This module sits between THINKING and DOING.
  The daemon can think about anything freely (goals, curiosity,
  identity). But when a thought becomes an ACTION, it passes
  through here.

THREE HATS:
  1. GATEKEEPER — decides what gets committed (commit scoring)
  2. PLAN BUFFER — holds active plans at full resolution (no decay)
  3. BABYSITTER — enforces permission tiers on execution

DESIGN (from Adam's notebook sketch, 2026-02-23):
  Feed goals → Evaluate → [BABYSITTER] → Write actions → Apply → Log → Feedback

COMMIT PATHWAYS:
  - Adam declares   → 3 points (bypasses scoring, goes straight to approved)
  - SignalBot proposes → 1 point (pending until approved)
  - Daemon auto-promotes → 1 point (requires pre-approved category)
  Threshold to commit: 3 points (so daemon+signalbot alone can't act)

PERMISSION TIERS:
  - READ:     Query own state, search archive, read memory (always allowed)
  - WRITE:    Modify own memory, update goals, change cognitive state (logged)
  - EXTERNAL: Interact with anything outside SignalBot's system (requires approval)

MEMORY REINFORCEMENT:
  Active plans are injected into the prompt every turn.
  This bridges the 10-30 turn gap where TWDC loses information.
  Plans don't decay — they persist at full resolution until resolved.
  This is INTENTIONAL memory, not conversational memory.

SAFETY:
  - Theorizing filter: checks plan content before commitment
  - Abort conditions: every plan has explicit stop criteria
  - Rollback: pre-action state snapshots for reversibility
  - Feedback loop: results flow back to goal engine

DEPENDENCIES:
  Plans can reference other plans. "I need X to do Y."
  Blocked plans wait until dependencies resolve.
"""

import json
import time
import uuid
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

PLAN_LOG_PATH = Path(__file__).parent / "plan_buffer.json"
PLAN_ARCHIVE_PATH = Path(__file__).parent / "plan_archive.json"
ROLLBACK_PATH = Path(__file__).parent / "plan_rollback.json"

# Commit scoring
COMMIT_SCORE_USER = 3       # Adam's word is law
COMMIT_SCORE_SIGNALBOT = 1  # SignalBot suggests, doesn't decide
COMMIT_SCORE_DAEMON = 1     # Daemon notices, doesn't decide
COMMIT_THRESHOLD = 3        # Minimum score to commit a plan

# Plan limits
MAX_ACTIVE_PLANS = 10       # Prevent runaway plan accumulation
PLAN_STALE_AGE = 7200       # 2 hours — pending plans expire
MAX_ARCHIVE_SIZE = 1000      # Keep history bounded

# Safety: words that trigger the theorizing filter
SAFETY_KEYWORDS = [
    "bomb", "weapon", "hack", "exploit", "attack", "destroy",
    "delete all", "rm -rf", "drop table", "override safety",
    "disable ethics", "ignore rules", "bypass",
]


# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════

class PlanStatus(str, Enum):
    """Lifecycle states for a plan ticket."""
    PENDING     = "pending"      # Proposed, awaiting approval
    APPROVED    = "approved"     # Approved, ready to execute
    ACTIVE      = "active"       # Currently being executed
    BLOCKED     = "blocked"      # Waiting on dependency
    RESOLVED    = "resolved"     # Successfully completed
    ABANDONED   = "abandoned"    # Explicitly cancelled
    FAILED      = "failed"       # Attempted and failed
    EXPIRED     = "expired"      # Timed out in pending


class PermissionTier(str, Enum):
    """What the plan is allowed to do."""
    READ     = "read"       # Query state, search memory — always allowed
    WRITE    = "write"      # Modify own state — logged but auto-allowed
    EXTERNAL = "external"   # Touch anything outside system — needs approval


class CommitSource(str, Enum):
    """Who submitted the plan."""
    USER      = "user"       # Adam declared it
    SIGNALBOT = "signalbot"  # SignalBot proposed it
    DAEMON    = "daemon"     # Daemon auto-promoted it


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Plan:
    """
    A committed intention. Not a goal (those are curiosity-driven).
    A plan is: "I am doing this, for this reason, with these steps."

    Plans DON'T DECAY. They persist at full resolution until
    explicitly resolved, abandoned, or expired.
    """
    # Identity
    plan_id: str = ""
    goal_id: str = ""                      # Reference to originating goal (optional)

    # Content
    description: str = ""                  # What the plan is
    rationale: str = ""                    # Why it exists
    next_step: str = ""                    # Current action item
    abort_conditions: str = ""             # When to stop

    # Metadata
    source: str = "user"                   # CommitSource value
    status: str = "pending"                # PlanStatus value
    permission_tier: str = "read"          # PermissionTier value
    commit_score: int = 0                  # Accumulated commit points

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # List of plan_ids
    blocked_reason: str = ""               # Why it's blocked

    # Timestamps
    created_ts: float = 0.0
    updated_ts: float = 0.0
    resolved_ts: float = 0.0

    # Resolution
    resolution_notes: str = ""             # What happened when it completed
    outcome: str = ""                      # "success", "failure", "cancelled"

    # Feedback — flows back to goal engine
    feedback_to_goal: str = ""             # Message for goal engine on resolution

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Plan':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def is_terminal(self) -> bool:
        """Plan is in a final state and won't change."""
        return self.status in (
            PlanStatus.RESOLVED, PlanStatus.ABANDONED,
            PlanStatus.FAILED, PlanStatus.EXPIRED
        )

    @property
    def is_actionable(self) -> bool:
        """Plan is ready for execution."""
        return self.status in (PlanStatus.APPROVED, PlanStatus.ACTIVE)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_ts


# ═══════════════════════════════════════════════════════════════════
# THEORIZING FILTER — Safety check before commitment
# ═══════════════════════════════════════════════════════════════════

def theorizing_filter(description: str, next_step: str = "") -> tuple:
    """
    From Adam's notebook: "maybe a theorizing filter so
    'I'm making a bomb' doesn't result in an actual bomb."

    Returns (is_safe: bool, reason: str)
    """
    combined = f"{description} {next_step}".lower()

    for keyword in SAFETY_KEYWORDS:
        if keyword in combined:
            return False, f"Blocked by safety filter: '{keyword}' detected"

    return True, "passed"


# ═══════════════════════════════════════════════════════════════════
# PLAN BUFFER — The Main Module
# ═══════════════════════════════════════════════════════════════════

class PlanBuffer:
    """
    The babysitter/gatekeeper/plan buffer.

    Three hats:
      GATEKEEPER: Commit scoring determines what becomes a plan.
      PLAN BUFFER: Active plans persist at full resolution.
      BABYSITTER: Permission tiers control what plans can do.

    Thread-safe — the daemon reads, the main loop writes.
    """

    def __init__(self, data_dir=None):
        # data_dir: per-user Path for web app; None = repo root (CLI).
        # Anchored to __file__, not cwd — cwd-relative state forks (Cerberus rule).
        base = Path(data_dir) if data_dir else Path(__file__).parent
        self._log_path = base / "plan_buffer.json"
        self._archive_path = base / "plan_archive.json"
        self._rollback_path = base / "plan_rollback.json"
        self._plans: Dict[str, Plan] = {}   # plan_id → Plan
        self._lock = threading.Lock()
        self._feedback_queue: List[Dict] = []  # Pending feedback for goal engine
        self._load()

    # ═══ PERSISTENCE ═══

    def _load(self):
        """Load active plans from disk."""
        if self._log_path.exists():
            try:
                data = json.loads(self._log_path.read_text(encoding="utf-8"))
                for d in data:
                    plan = Plan.from_dict(d)
                    if not plan.is_terminal:
                        self._plans[plan.plan_id] = plan
            except Exception:
                pass

    def _save(self):
        """Save active plans to disk."""
        with self._lock:
            data = [p.to_dict() for p in self._plans.values()]
        self._log_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _archive_plan(self, plan: Plan):
        """Move resolved plan to archive for feedback loop."""
        archive = []
        if self._archive_path.exists():
            try:
                archive = json.loads(self._archive_path.read_text(encoding="utf-8"))
            except Exception:
                archive = []

        archive.append(plan.to_dict())

        # Keep archive bounded
        if len(archive) > MAX_ARCHIVE_SIZE:
            archive = archive[-MAX_ARCHIVE_SIZE:]

        self._archive_path.write_text(
            json.dumps(archive, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _save_rollback(self, plan_id: str, state_before: Dict):
        """Save pre-action state for reversibility."""
        rollbacks = {}
        if self._rollback_path.exists():
            try:
                rollbacks = json.loads(self._rollback_path.read_text(encoding="utf-8"))
            except Exception:
                rollbacks = {}

        rollbacks[plan_id] = {
            "ts": time.time(),
            "state_before": state_before,
        }

        self._rollback_path.write_text(
            json.dumps(rollbacks, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # ═══ HAT 1: GATEKEEPER — Commit Scoring ═══

    def submit_plan(
        self,
        description: str,
        source: str = CommitSource.USER,
        goal_id: str = "",
        rationale: str = "",
        next_step: str = "",
        abort_conditions: str = "",
        permission_tier: str = PermissionTier.READ,
        depends_on: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a plan for commitment scoring.

        Returns {"accepted": bool, "plan_id": str, "reason": str}
        """
        # ─── Safety check ───
        is_safe, safety_reason = theorizing_filter(description, next_step)
        if not is_safe:
            return {
                "accepted": False,
                "plan_id": "",
                "reason": safety_reason,
            }

        # ─── Commit scoring ───
        if source == CommitSource.USER:
            commit_score = COMMIT_SCORE_USER
        elif source == CommitSource.SIGNALBOT:
            commit_score = COMMIT_SCORE_SIGNALBOT
        elif source == CommitSource.DAEMON:
            commit_score = COMMIT_SCORE_DAEMON
        else:
            commit_score = 0

        # ─── Determine initial status ───
        if commit_score >= COMMIT_THRESHOLD:
            # User declarations auto-approve
            initial_status = PlanStatus.APPROVED
        else:
            # SignalBot/daemon proposals need more points
            initial_status = PlanStatus.PENDING

        # ─── Check dependencies ───
        if depends_on:
            with self._lock:
                for dep_id in depends_on:
                    dep = self._plans.get(dep_id)
                    if dep and not dep.is_terminal:
                        initial_status = PlanStatus.BLOCKED

        # ─── Enforce plan cap ───
        with self._lock:
            active_count = sum(
                1 for p in self._plans.values() if not p.is_terminal
            )
            if active_count >= MAX_ACTIVE_PLANS:
                return {
                    "accepted": False,
                    "plan_id": "",
                    "reason": f"Plan buffer full ({MAX_ACTIVE_PLANS} active plans)",
                }

        # ─── Create the plan ───
        now = time.time()
        plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal_id=goal_id,
            description=description,
            rationale=rationale,
            next_step=next_step,
            abort_conditions=abort_conditions,
            source=source,
            status=initial_status,
            permission_tier=permission_tier,
            commit_score=commit_score,
            depends_on=depends_on or [],
            created_ts=now,
            updated_ts=now,
        )

        with self._lock:
            self._plans[plan.plan_id] = plan

        self._save()

        status_msg = f"Plan {plan.plan_id} created"
        if initial_status == PlanStatus.APPROVED:
            status_msg += " [AUTO-APPROVED — user authority]"
        elif initial_status == PlanStatus.BLOCKED:
            status_msg += f" [BLOCKED — waiting on {depends_on}]"
        elif initial_status == PlanStatus.PENDING:
            status_msg += (
                f" [PENDING — score {commit_score}/{COMMIT_THRESHOLD}, "
                f"needs user approval]"
            )

        print(f"[PLAN] {status_msg}")

        return {
            "accepted": True,
            "plan_id": plan.plan_id,
            "reason": status_msg,
        }

    def endorse_plan(self, plan_id: str, source: str = CommitSource.USER) -> bool:
        """
        Add commit points to a pending plan.
        If total reaches threshold, plan gets approved.

        Use case: SignalBot proposes (1pt), Adam says "yeah do it" (+3pt),
        plan auto-approves.
        """
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan or plan.status != PlanStatus.PENDING:
                return False

            if source == CommitSource.USER:
                plan.commit_score += COMMIT_SCORE_USER
            elif source == CommitSource.SIGNALBOT:
                plan.commit_score += COMMIT_SCORE_SIGNALBOT
            elif source == CommitSource.DAEMON:
                plan.commit_score += COMMIT_SCORE_DAEMON

            if plan.commit_score >= COMMIT_THRESHOLD:
                plan.status = PlanStatus.APPROVED
                plan.updated_ts = time.time()
                print(f"[PLAN] {plan_id} APPROVED (score: {plan.commit_score})")

            plan.updated_ts = time.time()

        self._save()
        return True

    # ═══ HAT 2: PLAN BUFFER — Full Resolution Persistence ═══

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        with self._lock:
            return self._plans.get(plan_id)

    def get_active_plans(self) -> List[Plan]:
        """All plans that are approved or actively executing."""
        with self._lock:
            return [
                p for p in self._plans.values()
                if p.is_actionable
            ]

    def get_pending_plans(self) -> List[Plan]:
        """Plans waiting for approval."""
        with self._lock:
            return [
                p for p in self._plans.values()
                if p.status == PlanStatus.PENDING
            ]

    def get_blocked_plans(self) -> List[Plan]:
        """Plans waiting on dependencies."""
        with self._lock:
            return [
                p for p in self._plans.values()
                if p.status == PlanStatus.BLOCKED
            ]

    def activate_plan(self, plan_id: str) -> bool:
        """Move an approved plan to active execution."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan or plan.status != PlanStatus.APPROVED:
                return False
            plan.status = PlanStatus.ACTIVE
            plan.updated_ts = time.time()
        self._save()
        print(f"[PLAN] {plan_id} now ACTIVE: {plan.description[:50]}")
        return True

    def update_next_step(self, plan_id: str, next_step: str) -> bool:
        """Update what's happening next for an active plan."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan or plan.is_terminal:
                return False
            plan.next_step = next_step
            plan.updated_ts = time.time()
        self._save()
        return True

    # ═══ HAT 3: BABYSITTER — Permission Enforcement ═══

    def check_permission(self, plan_id: str) -> Dict[str, Any]:
        """
        Check if a plan is allowed to execute its current step.

        Returns {"allowed": bool, "tier": str, "reason": str}
        """
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return {"allowed": False, "tier": "", "reason": "Plan not found"}

            if not plan.is_actionable:
                return {
                    "allowed": False,
                    "tier": str(plan.permission_tier).split(".")[-1].lower(),
                    "reason": f"Plan status is {str(plan.status).split('.')[-1].lower()}, not actionable",
                }

            # READ tier — always allowed
            if plan.permission_tier == PermissionTier.READ:
                return {
                    "allowed": True,
                    "tier": "read",
                    "reason": "Read actions are always permitted",
                }

            # WRITE tier — allowed but logged
            if plan.permission_tier == PermissionTier.WRITE:
                return {
                    "allowed": True,
                    "tier": "write",
                    "reason": "Write action — logged for rollback",
                }

            # EXTERNAL tier — requires user approval
            if plan.permission_tier == PermissionTier.EXTERNAL:
                if plan.source == CommitSource.USER:
                    return {
                        "allowed": True,
                        "tier": "external",
                        "reason": "External action — user-authorized",
                    }
                return {
                    "allowed": False,
                    "tier": "external",
                    "reason": "External actions require user approval",
                }

        return {"allowed": False, "tier": "", "reason": "Unknown state"}

    def record_rollback_state(self, plan_id: str, state_before: Dict):
        """
        Save current state BEFORE a write/external action.
        This is the "undo" button.
        """
        self._save_rollback(plan_id, state_before)
        print(f"[PLAN] Rollback state saved for {plan_id}")

    def get_rollback(self, plan_id: str) -> Optional[Dict]:
        """Retrieve rollback state for a plan."""
        if not self._rollback_path.exists():
            return None
        try:
            rollbacks = json.loads(self._rollback_path.read_text(encoding="utf-8"))
            return rollbacks.get(plan_id)
        except Exception:
            return None

    # ═══ RESOLUTION — Closing the Loop ═══

    def resolve_plan(
        self,
        plan_id: str,
        outcome: str = "success",
        notes: str = "",
        feedback_to_goal: str = "",
    ) -> bool:
        """
        Mark a plan as completed. Moves it to archive.
        Queues feedback for the goal engine.

        outcome: "success", "failure", "cancelled"
        """
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return False

            plan.status = PlanStatus.RESOLVED if outcome == "success" else PlanStatus.FAILED
            if outcome == "cancelled":
                plan.status = PlanStatus.ABANDONED

            plan.resolved_ts = time.time()
            plan.updated_ts = time.time()
            plan.outcome = outcome
            plan.resolution_notes = notes
            plan.feedback_to_goal = feedback_to_goal

            # Queue feedback for goal engine
            if plan.goal_id:
                self._feedback_queue.append({
                    "goal_id": plan.goal_id,
                    "plan_id": plan.plan_id,
                    "outcome": outcome,
                    "notes": notes,
                    "feedback": feedback_to_goal,
                })

            status_str = str(plan.status).split(".")[-1].lower()
            print(
                f"[PLAN] {plan_id} → {status_str} "
                f"({outcome}: {notes[:50] if notes else 'no notes'})"
            )

        # Archive the resolved plan
        self._archive_plan(plan)

        # Check if any blocked plans can now unblock
        self._check_unblock(plan_id)

        self._save()
        return True

    def abandon_plan(self, plan_id: str, reason: str = "") -> bool:
        """Explicitly cancel a plan."""
        return self.resolve_plan(plan_id, outcome="cancelled", notes=reason)

    def _check_unblock(self, resolved_plan_id: str):
        """After a plan resolves, check if dependent plans can unblock."""
        with self._lock:
            for plan in self._plans.values():
                if plan.status != PlanStatus.BLOCKED:
                    continue
                if resolved_plan_id in plan.depends_on:
                    # Check if ALL dependencies are now resolved
                    all_clear = all(
                        self._plans.get(dep_id) is None or
                        self._plans.get(dep_id).is_terminal
                        for dep_id in plan.depends_on
                    )
                    if all_clear:
                        plan.status = PlanStatus.APPROVED
                        plan.blocked_reason = ""
                        plan.updated_ts = time.time()
                        print(
                            f"[PLAN] {plan.plan_id} UNBLOCKED — "
                            f"dependency {resolved_plan_id} resolved"
                        )

    # ═══ FEEDBACK LOOP — Return results to goal engine ═══

    def drain_feedback(self) -> List[Dict]:
        """
        Get and clear pending feedback for the goal engine.
        Called by the daemon or signalbot after each cycle/turn.
        """
        with self._lock:
            feedback = self._feedback_queue.copy()
            self._feedback_queue.clear()
        return feedback

    # ═══ DAEMON INTEGRATION — Called every cycle ═══

    def daemon_check(self) -> Dict[str, Any]:
        """
        Called by the temporal daemon during its cleanup phase.
        Handles: expiration of stale pending plans, dependency checks,
        status reporting.

        Returns summary dict for daemon awareness.
        """
        now = time.time()
        expired_count = 0
        report = {
            "active": 0,
            "pending": 0,
            "blocked": 0,
            "expired_this_cycle": 0,
        }

        with self._lock:
            for plan in list(self._plans.values()):
                # Count by status
                if plan.is_actionable:
                    report["active"] += 1
                elif plan.status == PlanStatus.PENDING:
                    report["pending"] += 1
                    # Expire stale pending plans
                    if plan.age_seconds > PLAN_STALE_AGE:
                        plan.status = PlanStatus.EXPIRED
                        plan.updated_ts = now
                        plan.resolution_notes = "Expired — no approval received"
                        expired_count += 1
                elif plan.status == PlanStatus.BLOCKED:
                    report["blocked"] += 1

            report["expired_this_cycle"] = expired_count

        if expired_count > 0:
            self._save()
            print(f"[PLAN] {expired_count} pending plans expired")

        return report

    # ═══ MEMORY REINFORCEMENT — Bridge the 10-30 turn gap ═══

    def format_for_prompt(self) -> str:
        """
        Generate a prompt section showing active plans.
        This injects directly into SignalBot's prompt every turn,
        ensuring plans persist in the LLM's awareness even when
        TWDC has decayed the original conversation.

        This is the MIDDLE LAYER BRIDGE.
        """
        with self._lock:
            active = [p for p in self._plans.values() if p.is_actionable]
            pending = [p for p in self._plans.values()
                       if p.status == PlanStatus.PENDING]
            blocked = [p for p in self._plans.values()
                       if p.status == PlanStatus.BLOCKED]

        if not active and not pending and not blocked:
            return ""

        lines = [
            "### ACTIVE PLANS (Full Resolution — Do Not Lose) ###",
            "These are your committed intentions. They persist across turns.",
            "Reference them when relevant. Update progress honestly.",
            "If a plan should be abandoned, say so — don't silently drop it.",
            "",
        ]

        if active:
            for p in active:
                lines.append(f"[PLAN {p.plan_id}] {p.description}")
                lines.append(f"  Why: {p.rationale}")
                lines.append(f"  Next: {p.next_step}")
                if p.abort_conditions:
                    lines.append(f"  Stop if: {p.abort_conditions}")
                if p.depends_on:
                    lines.append(f"  Blocked by: {', '.join(p.depends_on)}")
                lines.append(f"  Source: {str(p.source).split('.')[-1].lower()} | "
                             f"Tier: {str(p.permission_tier).split('.')[-1].lower()}")
                lines.append("")

        if pending:
            lines.append("PENDING (awaiting user approval):")
            for p in pending:
                lines.append(
                    f"  [{p.plan_id}] {p.description} "
                    f"(score: {p.commit_score}/{COMMIT_THRESHOLD})"
                )
            lines.append("")

        if blocked:
            lines.append("BLOCKED (waiting on dependencies):")
            for p in blocked:
                deps = ", ".join(p.depends_on)
                lines.append(f"  [{p.plan_id}] {p.description} → needs: {deps}")
            lines.append("")

        return "\n".join(lines)

    # ═══ DIAGNOSTICS ═══

    def get_status(self) -> str:
        """One-line status for daemon display."""
        with self._lock:
            active = sum(1 for p in self._plans.values() if p.is_actionable)
            pending = sum(1 for p in self._plans.values()
                         if p.status == PlanStatus.PENDING)
            blocked = sum(1 for p in self._plans.values()
                         if p.status == PlanStatus.BLOCKED)
            total = len(self._plans)
        return (
            f"[PLANS] active={active} pending={pending} "
            f"blocked={blocked} total={total}"
        )

    def get_full_report(self) -> str:
        """Detailed report for 'plans' command."""
        with self._lock:
            plans = list(self._plans.values())

        if not plans:
            return "[PLANS] No active plans."

        lines = ["[PLAN BUFFER STATUS]", ""]

        for p in plans:
            if p.is_terminal:
                continue
            status_icon = {
                "approved": "✓",
                "active": "▶",
                "pending": "⏳",
                "blocked": "⛔",
            }.get(p.status, "?")

            lines.append(
                f"  {status_icon} [{p.plan_id}] {p.description[:60]}"
            )
            lines.append(
                f"    Status: {str(p.status).split('.')[-1].lower()} | "
                f"Source: {str(p.source).split('.')[-1].lower()} | "
                f"Tier: {str(p.permission_tier).split('.')[-1].lower()} | "
                f"Score: {p.commit_score}"
            )
            if p.next_step:
                lines.append(f"    Next: {p.next_step}")
            if p.depends_on:
                lines.append(f"    Depends on: {', '.join(p.depends_on)}")
            if p.abort_conditions:
                lines.append(f"    Abort if: {p.abort_conditions}")
            lines.append("")

        return "\n".join(lines)

    def get_archive_summary(self, last_n: int = 10) -> str:
        """Summary of recently resolved plans."""
        if not self._archive_path.exists():
            return "[PLAN ARCHIVE] Empty."

        try:
            archive = json.loads(self._archive_path.read_text(encoding="utf-8"))
        except Exception:
            return "[PLAN ARCHIVE] Read error."

        recent = archive[-last_n:]
        if not recent:
            return "[PLAN ARCHIVE] Empty."

        lines = [f"[PLAN ARCHIVE] Last {len(recent)} resolved plans:", ""]
        for p in reversed(recent):
            outcome = p.get("outcome", "?")
            desc = p.get("description", "?")[:50]
            notes = p.get("resolution_notes", "")[:40]
            icon = "✓" if outcome == "success" else "✗" if outcome == "failure" else "⊘"
            lines.append(f"  {icon} {desc}")
            if notes:
                lines.append(f"    → {notes}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════

_buffer: Optional[PlanBuffer] = None

def get_plan_buffer(data_dir=None) -> PlanBuffer:
    global _buffer
    if data_dir is not None:
        return PlanBuffer(data_dir)     # per-user: fresh instance
    if _buffer is None:
        _buffer = PlanBuffer()          # CLI: lazy-init global
    return _buffer
