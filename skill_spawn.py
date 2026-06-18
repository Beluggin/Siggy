#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
SKILL SPAWN — Lane 2: orchestrator–worker spawn executor
═══════════════════════════════════════════════════════════════════

The second lane of the initiative dispatcher: a goal becomes the
CONTRACT for a short-lived worker agent (headless Claude Code) that
runs in an isolated git worktree and settles through the existing
liveness bus. No new receipt machinery — gate/settle one altitude up.

THE LOOP (same shape as a tank command):
    contract goes out as INTENT          (negator=true)
    worker runs in isolation             (the world acts)
    SPAWN_RESULT.json + check command    (the world's verdict)
    publish_settle(goal_id, done|false)  (daemons react via the bus)

ISOLATION DOCTRINE (deliberate Cerberus INVERSION):
  - git worktree = tracked files only, own branch, shares git objects
    (disk-cheap). The child CANNOT touch the parent's state files —
    they're untracked, so they don't even exist in the worktree.
  - SPAWN_MODE=1 in the child's env. Explicit spawn mode, never
    accidental — that's the rule that keeps split-brain dead.
  - Deliverables stay on the spawn branch for YOUR review. Nothing
    merges back automatically.

GUARDRAIL DOCTRINE (same as lane 1 — caps live at the executor):
  - one spawn at a time (lockfile)
  - daily cap, persisted across restarts
  - wall-clock timeout per spawn (kill + settle false)
  - --max-turns on the child
  - disk floor check (same 1.5GB rule as freeze_snapshot)
  - no SPAWN_RESULT.json = FAIL. A worker that crashes without
    settling settles false — fail-safe default, same as the leash.

MANUAL FIRST: drive it from the CLI until the loop is boring, THEN
wire the dispatcher's explore/resolve candidates to maybe_spawn().

    python3 skill_spawn.py --goal-id fix-foo \
        --contract "Fix the bug in foo.py: ..." \
        --check "python3 test_foo.py"
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from liveness import publish_settle, DONE, FAIL

# ─── Tunables ───
MODEL             = "sonnet"   # default worker model; opus for hard contracts
MAX_TURNS         = 25         # child agent loop cap
TIMEOUT_SECONDS   = 900        # 15 min wall clock, then kill + settle false
MAX_SPAWNS_PER_DAY = 3
MIN_FREE_GB       = 1.5        # same floor as freeze_snapshot
# What the worker may do inside its worktree. Tight on purpose —
# widen per-contract if a skill genuinely needs more.
ALLOWED_TOOLS = "Read,Edit,Write,Glob,Grep,Bash(python3 *)"

RESULT_FILE = "SPAWN_RESULT.json"

# The contract wrapper. The goal text is the body; this is the frame
# that makes the worker settle explicitly instead of just trailing off.
CONTRACT_TEMPLATE = """You are a spawned worker agent for SignalBot. You run in an
isolated git worktree — work ONLY inside this directory.

YOUR CONTRACT (complete this one goal, nothing else):
{contract}

{check_clause}
WHEN FINISHED you MUST write a file named {result_file} in the
working directory root containing exactly:
  {{"status": "done", "summary": "<one sentence: what you did and how you verified it>"}}
or, if you could not complete the contract:
  {{"status": "false", "summary": "<one sentence: what blocked you>"}}

Do not ask questions — there is no user. If blocked, settle false
with a useful summary. The summary is training data; make it honest.
"""

CHECK_CLAUSE = """SUCCESS CHECK (your work is verified by running this — make it pass):
  {check}
"""


class SpawnResult:
    def __init__(self, status: str, summary: str, telemetry: Dict):
        self.status = status          # liveness DONE | FAIL
        self.summary = summary
        self.telemetry = telemetry

    def __str__(self):
        return f"[{self.status}] {self.summary}"


class SkillSpawner:
    """One spawner per repo. State lives next to the other runtime files."""

    def __init__(self, repo_dir: Path = None, data_dir: Path = None):
        self.repo_dir = Path(repo_dir or Path(__file__).parent)
        self.data_dir = Path(data_dir or self.repo_dir)
        self.state_path = self.data_dir / "spawn_state.json"
        self.log_path = self.data_dir / "spawn_log.jsonl"
        self.lock_path = self.data_dir / "spawn.lock"
        self.spawned_today = 0
        self.spawn_day = self._today()
        self._load()

    # ═══ PUBLIC ═══

    def spawn(self, goal_id: str, contract: str,
              check: Optional[str] = None,
              model: str = MODEL,
              timeout: int = TIMEOUT_SECONDS) -> SpawnResult:
        """Run one worker for one goal. Blocks until settled."""
        gate_fail = self._gates()
        if gate_fail:
            # A refused spawn is a non-event, not a fail — same as the
            # gate's NO_FIRE. Nothing ran, so nothing settles.
            self._log({"event": "refused", "goal_id": goal_id, "why": gate_fail})
            return SpawnResult("NO_FIRE", gate_fail, {})

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch = f"spawn/{goal_id}-{ts}"
        worktree = self.repo_dir / "spawn_worktrees" / f"{goal_id}-{ts}"

        self.lock_path.write_text(str(os.getpid()))
        self.spawned_today += 1
        self._save()
        self._log({"event": "spawned", "goal_id": goal_id, "branch": branch,
                   "model": model, "contract": contract[:200]})

        prompt = CONTRACT_TEMPLATE.format(
            contract=contract,
            check_clause=CHECK_CLAUSE.format(check=check) if check else "",
            result_file=RESULT_FILE,
        )

        try:
            self._git("worktree", "add", "-b", branch, str(worktree))
            telemetry = self._run_child(prompt, worktree, model, timeout)
            status, summary = self._read_settle(worktree)

            # Trust but verify: the check command is the world's verdict,
            # and it outranks the worker's self-report.
            if check and status == DONE:
                ok = self._run_check(check, worktree)
                telemetry["check_passed"] = ok
                if not ok:
                    status, summary = FAIL, f"worker claimed done but check failed: {check}"

            self._harvest(worktree, branch, status)
        except Exception as e:
            status, summary, telemetry = FAIL, f"spawn machinery error: {e}", {}
            # Best-effort cleanup so a crashed spawn can't strand a worktree.
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                           cwd=self.repo_dir, capture_output=True)
        finally:
            self.lock_path.unlink(missing_ok=True)

        telemetry["summary"] = summary
        self._log({"event": "settled", "goal_id": goal_id, "status": status,
                   "summary": summary, "telemetry": telemetry, "branch": branch})

        # The settle bus — the daemons react, not us. done flips
        # Goal.unresolved via the registered adapter; false leaves it
        # open for the (future) exception loop. Same bus as the tank.
        publish_settle(goal_id, status, telemetry)
        return SpawnResult(status, summary, telemetry)

    # ═══ GATES ═══

    def _gates(self) -> Optional[str]:
        if self.lock_path.exists():
            return f"another spawn is running (lock: {self.lock_path})"
        if self.spawn_day != self._today():
            self.spawn_day, self.spawned_today = self._today(), 0
        if self.spawned_today >= MAX_SPAWNS_PER_DAY:
            return f"daily spawn cap reached ({MAX_SPAWNS_PER_DAY})"
        free_gb = shutil.disk_usage(self.repo_dir).free / 1e9
        if free_gb < MIN_FREE_GB:
            return f"disk floor: {free_gb:.1f}GB free < {MIN_FREE_GB}GB"
        return None

    # ═══ THE CHILD ═══

    def _run_child(self, prompt: str, worktree: Path, model: str,
                   timeout: int) -> Dict:
        """Headless Claude Code in the worktree. Isolated env: SPAWN_MODE
        set, cwd is the worktree, tools restricted to it."""
        env = dict(os.environ, SPAWN_MODE="1")
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt,
                 "--model", model,
                 "--max-turns", str(MAX_TURNS),
                 "--allowedTools", ALLOWED_TOOLS,
                 "--output-format", "json"],
                cwd=worktree, env=env, capture_output=True, text=True,
                timeout=timeout,
            )
            tele = {"wall_s": round(time.time() - t0, 1),
                    "exit_code": proc.returncode}
            # --output-format json gives cost + turn count for free —
            # that's the spawn-economics dataset.
            try:
                meta = json.loads(proc.stdout)
                tele["cost_usd"] = meta.get("total_cost_usd")
                tele["num_turns"] = meta.get("num_turns")
            except Exception:
                tele["stdout_tail"] = proc.stdout[-300:]
            return tele
        except subprocess.TimeoutExpired:
            return {"wall_s": round(time.time() - t0, 1), "timed_out": True}

    def _read_settle(self, worktree: Path):
        """The worker's self-report. Missing/garbled = FAIL (fail-safe)."""
        f = worktree / RESULT_FILE
        if not f.exists():
            return FAIL, "worker exited without writing SPAWN_RESULT.json"
        try:
            d = json.loads(f.read_text())
            status = DONE if d.get("status") == "done" else FAIL
            return status, d.get("summary", "(no summary)")
        except Exception:
            return FAIL, "SPAWN_RESULT.json unreadable"

    def _run_check(self, check: str, worktree: Path) -> bool:
        try:
            r = subprocess.run(check, shell=True, cwd=worktree,
                               capture_output=True, timeout=120)
            return r.returncode == 0
        except Exception:
            return False

    # ═══ HARVEST ═══

    def _harvest(self, worktree: Path, branch: str, status: str):
        """Commit the worker's output on its branch, then drop the
        worktree. The branch is the deliverable — review it, cherry-pick
        what's good, delete what isn't. Nothing auto-merges."""
        # The result file is the receipt, not the deliverable.
        (worktree / RESULT_FILE).unlink(missing_ok=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=worktree,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            subprocess.run(["git", "add", "-A"], cwd=worktree, capture_output=True)
            commit = subprocess.run(
                ["git", "commit", "-m",
                 f"spawn artifact ({status})\n\nCo-Authored-By: spawn worker"],
                cwd=worktree, capture_output=True, text=True)
            if commit.returncode != 0:
                # Never silently drop a deliverable — leave the worktree
                # in place so the work is recoverable by hand.
                print(f"[spawn] WARNING: commit failed, keeping worktree "
                      f"{worktree}: {commit.stderr.strip()[:200]}")
                return
            self._git("worktree", "remove", "--force", str(worktree))
            print(f"[spawn] deliverable on branch {branch} — "
                  f"review with: git diff master...{branch}")
        else:
            self._git("worktree", "remove", "--force", str(worktree))
            self._git("branch", "-D", branch)
            print(f"[spawn] no file changes; branch {branch} deleted")

    # ═══ PLUMBING ═══

    def _git(self, *args):
        subprocess.run(["git", *args], cwd=self.repo_dir, check=True,
                       capture_output=True, text=True)

    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _save(self):
        try:
            self.state_path.write_text(json.dumps(
                {"spawn_day": self.spawn_day, "spawned_today": self.spawned_today}))
        except Exception:
            pass

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            d = json.loads(self.state_path.read_text())
            self.spawn_day = d.get("spawn_day", self._today())
            self.spawned_today = d.get("spawned_today", 0)
        except Exception:
            pass

    def _log(self, row: Dict):
        row = {"ts": time.time(), **row}
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass


# ═══ LATER: the dispatcher hook (lane-2 autonomy — wire AFTER manual
#     runs are boring). explore/resolve candidates map to spawns; the
#     dispatcher's own gates (threshold, quiet hours) already ran. ═══

def maybe_spawn(candidate: Dict, spawner: SkillSpawner) -> Optional[SpawnResult]:
    if candidate.get("action_type") not in ("explore", "resolve"):
        return None
    return spawner.spawn(goal_id=candidate["goal_id"],
                         contract=candidate["description"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Spawn one worker agent for one goal.")
    ap.add_argument("--goal-id", required=True)
    ap.add_argument("--contract", required=True, help="what the worker must do")
    ap.add_argument("--check", help="shell command that must pass to settle done")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the contract prompt and exit (no spawn)")
    a = ap.parse_args()

    if a.dry_run:
        print(CONTRACT_TEMPLATE.format(
            contract=a.contract,
            check_clause=CHECK_CLAUSE.format(check=a.check) if a.check else "",
            result_file=RESULT_FILE))
        raise SystemExit(0)

    s = SkillSpawner()
    res = s.spawn(a.goal_id, a.contract, check=a.check,
                  model=a.model, timeout=a.timeout)
    print(res)
    raise SystemExit(0 if res.status == DONE else 1)
