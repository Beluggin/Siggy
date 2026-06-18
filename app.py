"""
SignalBot Home Base — Flask App (Multi-User Edition)
=====================================================
Now uses user_manager.py for:
  - Per-user authentication (hashed passphrases)
  - Per-user memory namespaces
  - Admin god mode panel
  - User management (add, ban, reset passphrase)
"""

import os
import json
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from functools import wraps
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, abort, send_file,
    Response, stream_with_context
)
import threading
from user_manager import get_user_manager, UserContext
from code_reader import get_code_context, get_file_list_brief, get_file_context

# Web search is optional — app runs fine without it
try:
    from web_search import web_search, news_search, format_search_for_prompt, format_news_for_prompt
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# Maps model key → human label + response_engine config
# ══════════════════════════════════════════════════════════════════

MODELS = {
    # Fable = new Claude tier above Opus, launched 2026-06-09 (was wired pre-launch as
    # "Mythos" against the rumoured string — now confirmed live as claude-fable-5).
    "fable":         {"label": "Claude Fable 5 (API)",          "use_anthropic": True,  "use_mistral": False, "anthropic_model": "claude-fable-5"},
    "opus48":        {"label": "Claude Opus 4.8 (API)",         "use_anthropic": True,  "use_mistral": False, "anthropic_model": "claude-opus-4-8"},
    "opus":          {"label": "Claude Opus 4.7 (API)",         "use_anthropic": True,  "use_mistral": False, "anthropic_model": "claude-opus-4-7"},
    "sonnet":        {"label": "Claude Sonnet 4.6 (API)",       "use_anthropic": True,  "use_mistral": False, "anthropic_model": "claude-sonnet-4-6"},
    "haiku":         {"label": "Claude Haiku 4.5 (API)",        "use_anthropic": True,  "use_mistral": False, "anthropic_model": "claude-haiku-4-5-20251001"},
    "mistral_med":   {"label": "Mistral Medium (API)",          "use_anthropic": False, "use_mistral": True},
    "mistral7b":     {"label": "Mistral 7b (local)",            "use_anthropic": False, "use_mistral": False, "ollama_model": "mistral:7b"},
    "mistral7b_q8":  {"label": "Mistral 7b Q8 (local)",        "use_anthropic": False, "use_mistral": False, "ollama_model": "mistral:7b-instruct-q8_0"},
    "llama31":       {"label": "Llama 3.1 8b Q8 (local)",      "use_anthropic": False, "use_mistral": False, "ollama_model": "llama3.1:8b-instruct-q8_0"},
    "gemma2":        {"label": "Gemma2:9b (local)",             "use_anthropic": False, "use_mistral": False, "ollama_model": "gemma2:9b"},
    "phi3":          {"label": "Phi3 (local)",                  "use_anthropic": False, "use_mistral": False, "ollama_model": "phi3:latest"},
    "phi4":          {"label": "Phi4 Full (local)",             "use_anthropic": False, "use_mistral": False, "ollama_model": "phi4:latest"},
    "gemma4_12b":    {"label": "Gemma4:12b (local)",            "use_anthropic": False, "use_mistral": False, "ollama_model": "gemma4:12b"},
    # Google Gemini — verify model IDs at ai.google.dev/gemini-api/docs/models
    "gemini_pro":        {"label": "Gemini 2.5 Pro (API)",           "use_anthropic": False, "use_mistral": False, "use_gemini": True,   "gemini_model": "gemini-2.5-pro"},
    "gemini_flash":      {"label": "Gemini 2.5 Flash (API)",         "use_anthropic": False, "use_mistral": False, "use_gemini": True,   "gemini_model": "gemini-2.5-flash"},
    "gemini_35_flash":   {"label": "Gemini 3.5 Flash (API)",         "use_anthropic": False, "use_mistral": False, "use_gemini": True,   "gemini_model": "gemini-3.5-flash"},
    # DeepSeek — needs DEEPSEEK_API_KEY in .env (separate from GEMINI_API_KEY)
    "deepseek_r1":       {"label": "DeepSeek R1 (API)",              "use_anthropic": False, "use_mistral": False, "use_deepseek": True,  "deepseek_model": "deepseek-reasoner"},
    # OpenAI — needs OPENAI_KEY in .env
    "gpt55":             {"label": "ChatGPT 5.5 (API)",              "use_anthropic": False, "use_mistral": False, "use_openai": True,    "openai_model": "gpt-5.5"},
    "gpt5":              {"label": "GPT-5 (API)",                    "use_anthropic": False, "use_mistral": False, "use_openai": True,    "openai_model": "gpt-5"},
    "gpt41":             {"label": "GPT-4.1 (API)",                  "use_anthropic": False, "use_mistral": False, "use_openai": True,    "openai_model": "gpt-4.1"},
    "gpt4o":             {"label": "GPT-4o (API)",                   "use_anthropic": False, "use_mistral": False, "use_openai": True,    "openai_model": "gpt-4o"},
}

DEFAULT_MODEL = "mistral_med"

# Fix #1 (2026-06-11 audit): apply_model() mutates response_engine module
# globals, and Flask serves requests on multiple threads — two users chatting
# at once could swap each other's model mid-call. Every apply_model +
# generate_response pair must run under this lock, and call metadata must be
# captured inside it too (get_last_call_meta is also a shared global).
_llm_lock = threading.Lock()


def apply_model(model_key: str):
    """Configure response_engine globals for the given model key."""
    import response_engine
    cfg = MODELS.get(model_key) or MODELS[DEFAULT_MODEL]
    response_engine.USE_ANTHROPIC = cfg["use_anthropic"]
    response_engine.USE_MISTRAL   = cfg["use_mistral"]
    response_engine.USE_GEMINI    = cfg.get("use_gemini", False)
    response_engine.USE_DEEPSEEK  = cfg.get("use_deepseek", False)
    response_engine.USE_OPENAI    = cfg.get("use_openai", False)
    if cfg.get("ollama_model"):
        response_engine.OLLAMA_MODEL = cfg["ollama_model"]
    if cfg.get("anthropic_model"):
        response_engine.ANTHROPIC_MODEL = cfg["anthropic_model"]
    if cfg.get("gemini_model"):
        response_engine.GEMINI_MODEL = cfg["gemini_model"]
    if cfg.get("deepseek_model"):
        response_engine.DEEPSEEK_MODEL = cfg["deepseek_model"]
    if cfg.get("openai_model"):
        response_engine.OPENAI_MODEL = cfg["openai_model"]


# ══════════════════════════════════════════════════════════════════
# COMMAND RESPONSE FORMATTERS
# Mirror the CLI signalbot.py commands for use in the web chat.
# These return pre-formatted text — no LLM call needed.
# ══════════════════════════════════════════════════════════════════

def _fmt_state(cog: dict) -> str:
    """Format 13D cognitive state vector (mirrors CLI 'state' command)."""
    tone = cog.get("tone", {})
    return "\n".join([
        "[COGNITIVE STATE]",
        f"  Frustration:  {cog.get('frustration', 0):.2f}",
        f"  Curiosity:    {cog.get('curiosity', 0):.2f}",
        f"  Confidence:   {cog.get('confidence', 0):.2f}",
        f"  Engagement:   {cog.get('engagement', 0):.2f}",
        f"  Identity:     {cog.get('identity_adherence', 0):.2f}",
        f"  Cog Load:     {cog.get('cognitive_load', 0):.2f}",
        f"  Tone: P={tone.get('playful', 0):.2f} F={tone.get('formal', 0):.2f} "
        f"C={tone.get('concise', 0):.2f} W={tone.get('warm', 0):.2f}",
    ])


def _fmt_facts(facts_data: dict) -> str:
    """Format indelible facts (mirrors CLI 'facts' command)."""
    facts = facts_data.get("facts", []) if isinstance(facts_data, dict) else []
    if not facts:
        return "[No indelible facts yet]"
    lines = ["[INDELIBLE FACTS]"]
    for f in facts:
        locked = " [LOCKED]" if f.get("locked") else ""
        lines.append(f"  \u2014 {f.get('fact', '')}{locked}")
    return "\n".join(lines)


def _fmt_archive(archive: list) -> str:
    """Format archive stats (mirrors CLI 'archive' command)."""
    if not isinstance(archive, list):
        archive = []
    total_turns = sum(ep.get("turn_count", 0) for ep in archive)
    all_tags: set = set()
    for ep in archive:
        all_tags.update(ep.get("tags", []))
    lines = [
        "[ARCHIVE]",
        f"  Episodes: {len(archive)}",
        f"  Total turns archived: {total_turns}",
        f"  Unique tags: {len(all_tags)}",
    ]
    if archive:
        sorted_ep = sorted(archive, key=lambda e: e.get("ts_start", 0))
        try:
            oldest = datetime.fromtimestamp(sorted_ep[0].get("ts_start", 0)).strftime("%Y-%m-%d %H:%M")
            newest = datetime.fromtimestamp(sorted_ep[-1].get("ts_end", 0)).strftime("%Y-%m-%d %H:%M")
            lines += [f"  Oldest: {oldest}", f"  Newest: {newest}"]
        except Exception:
            pass
    return "\n".join(lines)


# Roblox "one-prompt game" builder \u2014 gated to these lowercase usernames only.
# Invisible to everyone else: both the command AND its help line are hidden, and
# a non-allowed user typing "roblox ..." just falls through to normal chat.
ROBLOX_USERS = {"adam", "mason", "sophie", "griff"}
# Game scripts run long \u2014 this lifts NUM_PREDICT (the chat-tuned output cap) for
# roblox generation only. See the roblox block in api_chat().
ROBLOX_NUM_PREDICT = 8192

WEB_COMMANDS_HELP = """\
[COMMANDS]
  state          \u2014 cognitive state vectors
  facts          \u2014 learned indelible facts
  archive        \u2014 archive stats
  daemon         \u2014 daemon status + background thinking
  modes          \u2014 cognitive mode status + recent gaps
  curiosity      \u2014 curiosity signal + top goals
  plans          \u2014 plan buffer report
  dream on/off   \u2014 toggle [GROUND]/[DREAM] output tagging
  search <query> \u2014 web search (results injected into reply)
  news [topic]   \u2014 news search (results injected into reply)
  read code      \u2014 inject full codebase into context
  read code <f>  \u2014 inject specific file into context
  help           \u2014 this list\
"""

# Shown in `?`/help only for ROBLOX_USERS.
ROBLOX_HELP_LINE = "  roblox <idea>  \u2014 describe your game \u2192 paste-ready Roblox Luau (no SignalBot memory)"


def commands_help_for(username):
    """Help text, with the Roblox builder line added only for allowed users
    and the read-code lines shown only to admins (the command is admin-gated)."""
    help_text = WEB_COMMANDS_HELP
    if username in ROBLOX_USERS:
        # Slot the roblox line just above 'read code', mirroring the old order.
        help_text = help_text.replace(
            "  read code      \u2014",
            ROBLOX_HELP_LINE + "\n  read code      \u2014",
            1,
        )
    profile = user_mgr.get_user(username)
    if not profile or profile.role != "admin":
        help_text = "\n".join(
            line for line in help_text.splitlines()
            if not line.strip().startswith("read code")
        )
    return help_text


# ══════════════════════════════════════════════════════════════════
# PER-USER COGNITIVE SESSION
# Each user gets their own daemon + cognitive modules pointing at
# their data directory. Instances are created on first message and
# cached for the lifetime of the server process.
# ══════════════════════════════════════════════════════════════════

class UserCognition:
    """
    Holds all per-user cognitive system instances.
    Created once per user, cached in _user_sessions.
    The daemon runs as a background thread — started on first message.
    """

    def __init__(self, username: str, data_dir: Path):
        from cognitive_state import CognitiveStateEngine
        from indelible_facts import IndelibleFactsEngine
        from goal_engine_DAEMON import GoalEngine as DaemonGoalEngine
        from temporal_daemon import TemporalDaemon
        from memory_twdc_stateful import StatefulTWDCWrapper
        from cognitive_modes import CognitiveModeEngine
        from plan_buffer import PlanBuffer
        from initiative_dispatcher import InitiativeDispatcher

        self.username = username
        self.data_dir = data_dir

        # Create all cognitive instances pointing at the user's directory
        self.cog_state = CognitiveStateEngine(data_dir)
        self.indelible = IndelibleFactsEngine(data_dir)
        self.goal_engine = DaemonGoalEngine()
        self.mem_stateful = StatefulTWDCWrapper(data_dir)
        self.mode_engine = CognitiveModeEngine(data_dir)
        self.buf = PlanBuffer(data_dir)
        self.turn = 0

        # Daemon — started on first message, runs until server restarts
        self.daemon = TemporalDaemon(goal_engine=self.goal_engine, data_dir=data_dir)
        self._daemon_started = False
        self._lock = threading.Lock()

        # Lane 1 initiative dispatcher — the executor for the daemon's
        # ActionCandidate queue. Only polled if the user opted in.
        self.initiative = InitiativeDispatcher(data_dir)

    def ensure_daemon_running(self):
        """Start the daemon if it hasn't been started yet."""
        with self._lock:
            if not self._daemon_started:
                self.daemon.start()
                self._daemon_started = True


# Per-user session cache — keyed by username
_user_sessions: dict = {}
_sessions_lock = threading.Lock()


def get_user_cognition(username: str) -> UserCognition:
    """Get or create the cognitive session for a user."""
    with _sessions_lock:
        if username not in _user_sessions:
            profile = user_mgr.get_user(username)
            if not profile:
                return None
            data_dir = profile.data_dir
            data_dir.mkdir(parents=True, exist_ok=True)
            _user_sessions[username] = UserCognition(username, data_dir)
        return _user_sessions[username]


# ══════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════

app = Flask(__name__)
# Session-cookie signing key. Order: env var → persisted local file → generate once.
# Persisting it (vs a fresh random each boot) keeps logins alive across restarts.
# Only signs login cookies — bot memory/identity live on disk keyed by username,
# untouched by this. The file is gitignored so the key never enters the repo.
def _load_secret_key():
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    key_file = Path(__file__).parent / ".secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    import secrets
    key = secrets.token_hex(32)
    key_file.write_text(key)
    os.chmod(key_file, 0o600)  # owner-only, it's a credential
    return key

app.secret_key = _load_secret_key()

# Initialize user manager on startup
user_mgr = get_user_manager()

# If no users exist yet, redirect to first-run setup
FIRST_RUN = user_mgr.get_user_count() == 0


# ══════════════════════════════════════════════════════════════════
# AUTH DECORATORS
# ══════════════════════════════════════════════════════════════════

def login_required(f):
    """Require any authenticated user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Require admin (god mode) user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated


def get_current_user_context() -> UserContext:
    """
    Get the UserContext for the currently logged-in user.
    This tells SignalBot which data directory to use.
    """
    username = session.get("username")
    if not username:
        return None
    try:
        return UserContext(username, user_mgr)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════
# HELPER: Read user-specific data files
# ══════════════════════════════════════════════════════════════════

def read_user_json(username: str, filename: str, default=None):
    """Read a JSON file from a user's data directory."""
    data = user_mgr.read_user_data(username, filename)
    return data if data is not None else default


# ══════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    # If no users exist, redirect to first-run setup
    if user_mgr.get_user_count() == 0:
        return redirect(url_for("first_run"))

    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        passphrase = request.form.get("passphrase", "")

        profile = user_mgr.authenticate(name, passphrase)
        if profile:
            session["authenticated"] = True
            session["username"] = profile.username
            session["display_name"] = profile.display_name
            session["role"] = profile.role
            return redirect(url_for("dashboard"))

        # Check if banned
        sanitized = user_mgr._sanitize_username(name)
        user = user_mgr.get_user(sanitized)
        if user and user.is_banned:
            error = "This account has been suspended."
        else:
            error = "Wrong name or passphrase."

    return render_template("login.html", error=error)


@app.route("/first-run", methods=["GET", "POST"])
def first_run():
    """First-time setup: create admin account."""
    if user_mgr.get_user_count() > 0:
        return redirect(url_for("login"))

    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        passphrase = request.form.get("passphrase", "").strip()

        if not name or not passphrase:
            error = "Name and passphrase are both required."
        else:
            profile = user_mgr.setup_first_admin(name, passphrase)
            if profile:
                session["authenticated"] = True
                session["username"] = profile.username
                session["display_name"] = profile.display_name
                session["role"] = profile.role
                return redirect(url_for("dashboard"))
            else:
                error = "Setup failed. Check terminal for details."

    return render_template("first_run.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════════
# MAIN PAGES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        user=session.get("display_name", "User"),
        role=session.get("role", "user"),
        is_admin=session.get("role") == "admin",
    )


@app.route("/chat")
@login_required
def chat():
    username = session.get("username", "")
    profile = user_mgr.get_user(username)
    current_model = getattr(profile, "model", DEFAULT_MODEL)
    return render_template(
        "chat.html",
        user=session.get("display_name", "User"),
        role=session.get("role", "user"),
        tank_mode=session.get("tank_mode", False),
        initiative_on=getattr(profile, "initiative_enabled", False),
        models=MODELS,              # drive the dropdown off the registry (single source of truth)
        current_model=current_model,
    )


@app.route("/game")
@login_required
def game():
    return send_file(Path(__file__).parent / "flappy-mason.html")


@app.route("/candy-wars")
@login_required
def candy_wars():
    return send_file(Path(__file__).parent / "candy-wars.html")


# ══════════════════════════════════════════════════════════════════
# ADMIN PANEL (GOD MODE)
# ══════════════════════════════════════════════════════════════════

@app.route("/admin")
@admin_required
def admin_panel():
    """God mode: see all users and their stats."""
    users = user_mgr.get_all_users()
    user_stats = []
    for u in users:
        stats = user_mgr.get_user_stats(u.username)
        if stats:
            stats["model"] = getattr(u, "model", DEFAULT_MODEL)
            user_stats.append(stats)

    return render_template(
        "admin.html",
        user=session.get("display_name", "Admin"),
        user_stats=user_stats,
        models=MODELS,              # drive the per-user dropdown off the registry
    )


@app.route("/admin/user/<username>")
@admin_required
def admin_user_detail(username):
    """God mode: view a specific user's data."""
    stats = user_mgr.get_user_stats(username)
    if not stats:
        abort(404)

    # Load their recent conversations
    memory = read_user_json(username, "memory_log.json", [])
    recent_memory = memory[-20:] if isinstance(memory, list) else []

    # Load their cognitive state
    cog_state = read_user_json(username, "cognitive_state.json", {})

    # Load their indelible facts
    facts_data = read_user_json(username, "indelible_facts.json", {})
    facts = facts_data.get("facts", []) if isinstance(facts_data, dict) else []

    return render_template(
        "admin_user.html",
        user=session.get("display_name", "Admin"),
        target=stats,
        recent_memory=recent_memory,
        cog_state=cog_state,
        facts=facts,
    )


# ══════════════════════════════════════════════════════════════════
# ADMIN API ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route("/api/admin/add-user", methods=["POST"])
@admin_required
def api_admin_add_user():
    """Add a new user account."""
    data = request.get_json()
    name = data.get("name", "").strip()
    passphrase = data.get("passphrase", "").strip()
    role = data.get("role", "user")

    if not name or not passphrase:
        return jsonify({"error": "Name and passphrase required"}), 400

    if role not in ("user", "admin"):
        role = "user"

    profile = user_mgr.create_user(name, passphrase, role=role)
    if not profile:
        return jsonify({"error": "Username already taken or reserved"}), 400

    return jsonify({
        "ok": True,
        "username": profile.username,
        "display_name": profile.display_name,
        "role": profile.role,
    })


@app.route("/api/admin/set-role", methods=["POST"])
@admin_required
def api_admin_set_role():
    """Change a user's role (ban/unban/promote)."""
    data = request.get_json()
    username = data.get("username", "")
    new_role = data.get("role", "")

    if not username or new_role not in ("user", "admin", "banned"):
        return jsonify({"error": "Invalid username or role"}), 400

    # Don't let admin ban themselves
    if username == session.get("username") and new_role == "banned":
        return jsonify({"error": "You can't ban yourself"}), 400

    ok = user_mgr.set_role(username, new_role)
    if not ok:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"ok": True, "username": username, "role": new_role})


@app.route("/api/admin/reset-passphrase", methods=["POST"])
@admin_required
def api_admin_reset_passphrase():
    """Reset a user's passphrase."""
    data = request.get_json()
    username = data.get("username", "")
    new_passphrase = data.get("passphrase", "").strip()

    if not username or not new_passphrase:
        return jsonify({"error": "Username and new passphrase required"}), 400

    ok = user_mgr.change_passphrase(username, new_passphrase)
    if not ok:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"ok": True, "username": username})


@app.route("/api/admin/delete-user", methods=["POST"])
@admin_required
def api_admin_delete_user():
    """Delete a user account."""
    data = request.get_json()
    username = data.get("username", "")
    delete_data = data.get("delete_data", False)

    if not username:
        return jsonify({"error": "Username required"}), 400

    if username == session.get("username"):
        return jsonify({"error": "You can't delete yourself"}), 400

    ok = user_mgr.delete_user(username, delete_data=delete_data)
    if not ok:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"ok": True, "username": username})


@app.route("/api/set-model", methods=["POST"])
@login_required
def api_set_model():
    """Save the current user's model preference."""
    data = request.get_json()
    model_key = data.get("model", "")
    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400
    username = session.get("username", "")
    user_mgr.set_model(username, model_key)
    with _llm_lock:
        apply_model(model_key)
    return jsonify({"ok": True, "model": model_key, "label": MODELS[model_key]["label"]})


@app.route("/api/set-tank-mode", methods=["POST"])
@login_required
def api_set_tank_mode():
    """Toggle tank-lane routing for this session (the chat-window switch)."""
    data = request.get_json() or {}
    session["tank_mode"] = bool(data.get("on"))
    return jsonify({"ok": True, "tank_mode": session["tank_mode"]})


@app.route("/api/set-initiative", methods=["POST"])
@login_required
def api_set_initiative():
    """Per-user opt-in switch for daemon-initiated pings. Default OFF."""
    data = request.get_json() or {}
    username = session.get("username", "")
    enabled = bool(data.get("on"))
    user_mgr.set_initiative(username, enabled)
    return jsonify({"ok": True, "initiative": enabled})


def _make_initiative_renderer(cogn, username, user_model):
    """Build the talk-lane renderer the dispatcher calls when a candidate
    clears every guardrail. candidate dict → a real, forward-moving message
    in the bot's own voice (or None to skip).

    WHY this exists: the daemon captures goals straight from conversation, so
    a goal description is frequently a raw user line ("Ahhhh, way better").
    The old template wrapped that verbatim and pinged it back — the bot
    parroting the user at themselves. This renders an ACTUAL next thought:
    advance the thread, offer a fresh angle, or ask a sharp question — and is
    told in no uncertain terms not to quote the user. Same model + identity as
    the chat lane so the ping sounds like the same someone."""
    from initiative_dispatcher import build_initiative_prompt, DEFAULT_IDENTITY

    def render(candidate):
        thread = (candidate.get("description") or "").strip()
        if not thread:
            return None
        action = candidate.get("action_type", "think")
        reasoning = (candidate.get("reasoning") or "").strip()

        # Identity (voice) + recent conversation (so the ping can pursue the
        # actual next step, not float free of context).
        ctx = get_current_user_context()
        try:
            identity = ctx.get_path("signal_identity.txt").read_text(encoding="utf-8")
        except Exception:
            identity = DEFAULT_IDENTITY
        try:
            from memory_engine import MemoryEngine
            recent = MemoryEngine(cogn.data_dir).load_recent_memory(n=8)
            if recent == "(none)":
                recent = ""
        except Exception:
            recent = ""

        # Shared with initiative_swap_test.py — same prompt, both paths.
        prompt = build_initiative_prompt(identity, thread, action, reasoning, recent)

        try:
            from response_engine import generate_response
            with _llm_lock:
                apply_model(user_model)
                msg = generate_response(prompt)
        except Exception as e:
            print(f"[initiative] render failed: {e}")
            return None
        return msg
    return render


@app.route("/api/initiative")
@login_required
def api_initiative():
    """
    Lane 1 poll endpoint — the chat page hits this on a timer.
    Returns {"message": null} unless the user opted in, their daemon
    is running, AND a candidate clears every dispatcher guardrail.
    Deliberately never starts the daemon — polling must stay passive.
    """
    username = session.get("username", "")
    profile = user_mgr.get_user(username)
    if not profile or not getattr(profile, "initiative_enabled", False):
        return jsonify({"message": None})

    cogn = _user_sessions.get(username)
    if cogn is None or not cogn._daemon_started:
        return jsonify({"message": None})

    # The dispatcher owns selection + guardrails; the renderer (talk lane,
    # user's own model) owns the VOICE. Passing it in keeps the dispatcher
    # model-free and offline-testable while killing the parrot.
    user_model = getattr(profile, "model", DEFAULT_MODEL)
    renderer = _make_initiative_renderer(cogn, username, user_model)
    ping = cogn.initiative.poll(cogn.daemon, cogn.goal_engine, render_fn=renderer)
    if ping is None:
        return jsonify({"message": None})

    # The bot reached out unprompted — that counts to memory just like a chat
    # turn, so it remembers its own initiative (and TWDC/archive can score it).
    try:
        from memory_engine import MemoryEngine
        MemoryEngine(cogn.data_dir).save_initiative(
            ping.get("message", ""), cogn.cog_state.state.to_dict())
    except Exception as e:
        print(f"[initiative] memory save failed: {e}")

    return jsonify(ping)


@app.route("/api/admin/set-model", methods=["POST"])
@admin_required
def api_admin_set_model():
    """Set the LLM model for a specific user."""
    data = request.get_json()
    username = data.get("username", "")
    model_key = data.get("model", "")

    if not username:
        return jsonify({"error": "Username required"}), 400
    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    ok = user_mgr.set_model(username, model_key)
    if not ok:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "ok": True,
        "username": username,
        "model": model_key,
        "label": MODELS[model_key]["label"],
    })


@app.route("/api/admin/models")
@admin_required
def api_admin_models():
    """Return available models for the admin panel dropdown."""
    return jsonify({k: v["label"] for k, v in MODELS.items()})


@app.route("/api/admin/list-users")
@admin_required
def api_admin_list_users():
    """Get all users with stats (for admin panel refresh)."""
    users = user_mgr.get_all_users()
    result = []
    for u in users:
        stats = user_mgr.get_user_stats(u.username)
        if stats:
            stats["model"] = getattr(u, "model", DEFAULT_MODEL)
            result.append(stats)
    return jsonify({"users": result})


# ══════════════════════════════════════════════════════════════════
# USER API ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route("/api/status")
@login_required
def api_status():
    """
    Status endpoint — reads from the logged-in user's data.
    Returns real cognitive state, message counts, fact counts.
    """
    username = session.get("username", "")
    cog = read_user_json(username, "cognitive_state.json", {})
    memory = read_user_json(username, "memory_log.json", [])
    facts_data = read_user_json(username, "indelible_facts.json", {})
    archive = read_user_json(username, "memory_archive.json", [])

    # Counts
    msg_count = len(memory) if isinstance(memory, list) else 0
    fact_count = len(facts_data.get("facts", [])) if isinstance(facts_data, dict) else 0
    archive_count = len(archive) if isinstance(archive, list) else 0

    # Determine cognitive state label
    curiosity = cog.get("curiosity", 0)
    frustration = cog.get("frustration", 0)
    engagement = cog.get("engagement", 0)
    confidence = cog.get("confidence", 0)

    if frustration > 0.7:
        cog_label = "FRUSTRATED"
    elif curiosity > 0.7 and engagement > 0.5:
        cog_label = "CURIOUS"
    elif engagement > 0.7 and confidence > 0.5:
        cog_label = "ENGAGED"
    elif engagement < 0.3:
        cog_label = "IDLE"
    else:
        cog_label = "ACTIVE"

    # Build cognitive detail for the bar chart
    tone = cog.get("tone", {})
    cognitive_detail = {
        "curiosity": curiosity,
        "frustration": frustration,
        "confidence": confidence,
        "engagement": engagement,
        "identity_adherence": cog.get("identity_adherence", 0),
        "cognitive_load": cog.get("cognitive_load", 0),
    }
    # Add tone dimensions if they exist
    if isinstance(tone, dict):
        cognitive_detail["tone_playful"] = tone.get("playful", 0)
        cognitive_detail["tone_formal"] = tone.get("formal", 0)
        cognitive_detail["tone_concise"] = tone.get("concise", 0)
        cognitive_detail["tone_warm"] = tone.get("warm", 0)

    # Detect model from user profile
    user_model_key = getattr(user_mgr.get_user(username), "model", DEFAULT_MODEL)
    model_name = MODELS.get(user_model_key, MODELS[DEFAULT_MODEL])["label"]

    return jsonify({
        "bot_online": True,
        "messages_total": msg_count,
        "indelible_facts": fact_count,
        "archive_episodes": archive_count,
        "cognitive_state": cog_label,
        "cognitive_detail": cognitive_detail,
        "model": model_name,
        "user": session.get("display_name", "?"),
    })


def log_verbatim(username, text, source="web"):
    """Append raw user input VERBATIM (untouched — typos, grammar, casing all intact)
    for the personalized speech-pattern corpus harvest. Must never break the chat."""
    try:
        with open("verbatim_log.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.time(), "username": username,
                                "source": source, "text": text}) + "\n")
    except Exception:
        pass


# ── Tank bridge (OmniAddress → robot). TANK_URL points at the tank server;
# defaults to localhost:5000 (which is THIS app, so it safely no-ops until you
# set TANK_URL to the Pi at http://192.168.0.22:5000, or :5001 for local sim). ──
_tank_client = None
def get_tank_client():
    """Lazy TankClient singleton."""
    global _tank_client
    if _tank_client is None:
        from signalbot_tank_client import TankClient
        _tank_client = TankClient(os.environ.get("TANK_URL", "http://localhost:5000"))
    return _tank_client

def log_route(username, text, result):
    """Log the gate decision beside the raw input — CLARIFY/SAFE_STOP rows
    self-flag as misses for the harvest corpus. Never breaks the chat."""
    try:
        with open("tank_routes.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.time(), "username": username, "text": text,
                                "status": result.status, "lanes": result.lanes,
                                "tank_cmd": result.tank_cmd, "address": result.address,
                                "reason": result.reason}) + "\n")
    except Exception:
        pass


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """
    Chat endpoint — uses the logged-in user's data directory.
    """
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    user_image = data.get("image")  # base64 (no data: prefix) or None — vision path
    # Allow an image-only turn (no text); otherwise an empty message is still a 400.
    if not user_msg and not user_image:
        return jsonify({"error": "Empty message"}), 400

    # DEBUG: persist the EXACT bytes the server received (already phone-downscaled),
    # so we can run YOLO on what the model actually saw — not the original photo.
    # Overwrites one file → disk-safe. This is the "path for your camera pictures".
    if user_image:
        try:
            import base64
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "debug_last_upload.jpg"), "wb") as _f:
                _f.write(base64.b64decode(user_image))
            print("[upload-save] wrote debug_last_upload.jpg "
                  f"({len(user_image)} b64 chars)")
        except Exception as _e:
            print(f"[upload-save] failed: {_e}")

    username = session.get("username", "")
    msg_lower_raw = user_msg.lower().strip()
    is_roblox_gamebuild = (
        username in ROBLOX_USERS
        and (
            msg_lower_raw.startswith("roblox ")
            or msg_lower_raw.startswith("luau ")
            or msg_lower_raw.startswith("gamebuild ")
        )
    )
    if not is_roblox_gamebuild:
        log_verbatim(username, data.get("message", ""), source="web")  # raw, pre-strip — speech-pattern harvest

    # ── Command handling (mirrors CLI signalbot.py commands) ──
    # These short-circuit before the LLM call — no token cost.
    cmd = user_msg.lower().strip()
    search_injection = None  # may be set by search/news commands below

    def quick_reply(text):
        return jsonify({"reply": text, "timestamp": datetime.now().strftime("%H:%M:%S"), "sender": "SignalBot"})

    if cmd in ("help", "commands", "?"):
        return quick_reply(commands_help_for(username))

    if is_roblox_gamebuild:
        if cmd.startswith("roblox "):
            idea = user_msg[7:].strip()
        elif cmd.startswith("luau "):
            idea = user_msg[5:].strip()
        else:
            idea = user_msg[10:].strip()
        if not idea:
            return quick_reply("[ROBLOX] Usage: roblox <game mechanic or script idea>")
        try:
            user_model = getattr(user_mgr.get_user(username), "model", DEFAULT_MODEL)
            from roblox_luau_generator import generate_luau_bundle
            # Roblox mode: lift the chat output cap just for this call (game
            # scripts run long), then restore it no matter what. The whole
            # apply+bump+generate sequence holds _llm_lock so a concurrent
            # chat turn can't see the roblox cap or the wrong model.
            import response_engine
            with _llm_lock:
                apply_model(user_model)
                prev_cap = response_engine.NUM_PREDICT
                response_engine.NUM_PREDICT = ROBLOX_NUM_PREDICT
                try:
                    script_text, output_path = generate_luau_bundle(idea)
                finally:
                    response_engine.NUM_PREDICT = prev_cap
            saved = f"\n\n[SAVED] {output_path}" if output_path else ""
            return quick_reply(script_text + saved)
        except Exception as e:
            return quick_reply(f"[ROBLOX ERROR] {e}")

    ctx = get_current_user_context()
    if not ctx:
        return jsonify({"error": "Session expired. Please log in again."}), 401

    if cmd == "state":
        cog = read_user_json(username, "cognitive_state.json", {})
        return quick_reply(_fmt_state(cog))

    if cmd == "facts":
        facts_raw = read_user_json(username, "indelible_facts.json", {})
        return quick_reply(_fmt_facts(facts_raw))

    if cmd == "archive":
        archive_raw = read_user_json(username, "memory_archive.json", [])
        return quick_reply(_fmt_archive(archive_raw))

    if cmd == "dream on":
        session["dream_mode"] = True
        return quick_reply("[MODE] Dream mode ON — responses will use [GROUND]/[DREAM] tagging")

    if cmd == "dream off":
        session["dream_mode"] = False
        return quick_reply("[MODE] Dream mode OFF — [GROUND] only")

    if cmd in ("tank on", "tank mode on"):
        session["tank_mode"] = True
        return quick_reply("[TANK] Tank routing ON — motor commands (drive forward, turn left, stop…) drive the tank; everything else still chats. 'tank off' to disable.")

    if cmd in ("tank off", "tank mode off"):
        session["tank_mode"] = False
        return quick_reply("[TANK] Tank routing OFF")

    # Daemon/cognitive commands — use live per-user instances
    if cmd == "daemon":
        cogn = get_user_cognition(username)
        if cogn is None or not cogn._daemon_started:
            return quick_reply("[DAEMON] Not started yet — send a message first to boot it.")
        snap = cogn.daemon.get_snapshot()
        lines = [cogn.daemon.get_status(),
                 f"  Cycles since last msg: {snap.cycle_count}",
                 f"  Good Sense: {snap.good_sense:.2f}",
                 f"  Crap Threshold: {snap.crap_threshold:.2f}",
                 f"  Evaluated: {snap.items_evaluated}",
                 f"  Purged (total): {snap.items_purged}"]
        if snap.ambient_awareness:
            lines.append(f"  Ambient: {snap.ambient_awareness}")
        if snap.focus_summary:
            lines.append(f"  Focus: {snap.focus_summary}")
        if snap.top_recommendations:
            lines.append("  Top Recommendations:")
            for rec in snap.top_recommendations[:5]:
                lines.append(f"    [{rec['composite_score']:.2f}] {rec['action_type']}: "
                             f"{rec['description'][:50]}")
        return quick_reply("\n".join(lines))

    if cmd == "modes":
        cogn = get_user_cognition(username)
        if cogn is None or cogn.daemon._mode_engine is None:
            return quick_reply("[MODES] Mode engine not available.")
        me = cogn.daemon._mode_engine
        lines = [me.get_status()]
        for m in me.get_active_modes():
            if m.mode_id == 0:
                continue
            lines.append(f"  {m.name}: blend={m.blend_weight:.2f} activations={m.activation_count}")
        gaps = me.get_recent_gaps(3)
        if gaps:
            lines.append("  Recent gaps:")
            for g in gaps:
                lines.append(f"    {g['description']} — \"{g['user_input'][:40]}\"")
        return quick_reply("\n".join(lines))

    if cmd == "curiosity":
        cogn = get_user_cognition(username)
        if cogn is None:
            return quick_reply("[CURIOSITY] Session not available.")
        s = cogn.cog_state.state
        top_goals = cogn.goal_engine.get_top_curiosity_goals(3)
        lines = [
            "[CURIOSITY]",
            f"  State curiosity: {s.curiosity:.2f}",
            f"  Engagement:      {s.engagement:.2f}",
            f"  Identity:        {s.identity_adherence:.2f}",
        ]
        if top_goals:
            lines.append("  Top curiosity goals:")
            for g in top_goals:
                lines.append(f"    [{g.curiosity:.2f}] {g.description[:55]}")
        return quick_reply("\n".join(lines))

    if cmd == "plans":
        cogn = get_user_cognition(username)
        if cogn is None or cogn.daemon._plan_buffer is None:
            return quick_reply("[PLANS] Plan buffer not available.")
        return quick_reply(cogn.daemon._plan_buffer.get_full_report())

    # search/news fall through to LLM with results injected into the prompt
    if cmd.startswith("search ") and WEB_SEARCH_AVAILABLE:
        query = user_msg[7:].strip()
        if not query:
            return quick_reply("[SEARCH] No query provided")
        try:
            results = web_search(query)
            search_injection = format_search_for_prompt(results, query)
        except Exception as e:
            return quick_reply(f"[SEARCH ERROR] {e}")

    # bare "news" = top headlines; "news <topic>" = topic search. The trailing
    # space (or exact match) stops "newsletter"/"newsflash" from hijacking chat.
    if (cmd == "news" or cmd.startswith("news ")) and WEB_SEARCH_AVAILABLE:
        topic = user_msg[4:].strip()
        try:
            results = news_search(topic)
            search_injection = format_news_for_prompt(results, topic)
        except Exception as e:
            return quick_reply(f"[NEWS ERROR] {e}")

    # ── Tank lane routing (opt-in via the chat toggle / 'tank on') ──
    # Parse the message → OmniAddress → gate. A safe, routable motor command
    # drives the tank (optional 'for N seconds' → drive-then-auto-stop) and
    # short-circuits; talk-lane / non-commands fall through to the LLM. The
    # explicit text commands above always win. Never breaks chat on error.
    if session.get("tank_mode"):
        try:
            import omni_gate
            res = omni_gate.route(user_msg)
            log_route(username, data.get("message", ""), res)
            verb = (res.address or {}).get("verb")
            if res.status == "ROUTE" and "tank" in res.lanes and res.tank_cmd:
                tank = get_tank_client()
                if not tank.is_connected():
                    return quick_reply(f"[TANK] would {res.tank_cmd}, but tank unreachable — "
                                       f"safe no-op. Set TANK_URL to the robot (or :5001 sim).")
                tank.claim()
                dur = omni_gate.resolve_duration(user_msg)
                if dur and res.tank_cmd in ("forward", "backward", "left", "right"):
                    dur = min(dur, 10.0)  # safety cap — no runaway motion
                    # Once motion starts, the stop is non-negotiable: it lives
                    # in a finally with one retry, so an exception during the
                    # sleep (or a flaky first stop call) can't leave the robot
                    # driving off into the kitchen.
                    omni_gate.execute(res, tank)
                    try:
                        time.sleep(dur)
                    finally:
                        try:
                            tank.stop()
                        except Exception:
                            time.sleep(0.2)
                            tank.stop()   # one retry; if this raises, the outer
                                          # handler logs and the physical e-stop is on you
                    return quick_reply(f"[TANK] {res.tank_cmd} for {dur:g}s → stopped")
                omni_gate.execute(res, tank)
                return quick_reply(f"[TANK] {res.tank_cmd}")
            if res.status in ("SAFE_STOP", "CLARIFY") and verb in omni_gate.TANK_VERBS:
                return quick_reply(f"[TANK] {res.status}: {res.reason}")
            # talk-lane / non-motor → fall through to normal chat
        except Exception as e:
            # Routing must never break the chat — fall through to LLM. But say
            # so loudly: if this fired after motion started, the stop retry may
            # have failed and the tank could still be moving.
            print(f"[TANK] routing error (fell through to chat): {e}")

    # ── Get (or create) this user's cognitive session ──
    cogn = get_user_cognition(username)
    if cogn is None:
        return jsonify({"error": "Could not load cognitive session"}), 500
    cogn.ensure_daemon_running()

    # ── Pause daemon during inference (same as CLI pattern) ──
    cogn.daemon.pause()
    reply = "[Error: response not generated]"  # safe default if try block fails early

    try:
        # ── Daemon snapshot → inject background thinking into prompt ──
        daemon_snapshot = cogn.daemon.get_snapshot()
        daemon_cognition = daemon_snapshot.format_for_prompt(max_items=5)

        # ── Load user's recent + long-term memory ──
        from memory_engine import MemoryEngine
        mem_engine = MemoryEngine(cogn.data_dir)
        recent_text = mem_engine.load_recent_memory(n=12)
        if recent_text == "(none)":
            recent_text = ""

        # Wire daemon context into TWDC so memory re-scoring uses conversation bigrams
        cogn.mem_stateful.set_conversation_context(cogn.daemon._context, cogn.daemon._real_turns)
        long_memory = cogn.mem_stateful.build_long_memory_block_stateful(max_bullets=10)

        # ── Load identity ──
        identity_path = ctx.get_path("signal_identity.txt")
        try:
            identity = identity_path.read_text(encoding="utf-8")
        except Exception:
            identity = "You are SignalBot. Clever, candid, and slightly irreverent."

        # ── Indelible facts, tone, vitals, mode prompt ──
        facts_text = cogn.indelible.format_for_prompt(max_facts=20)
        tone = cogn.cog_state.get_tone_instructions()
        vitals = cogn.cog_state.get_vitals_report()

        # Mode resonance check — detect if active memory covers the question
        active_memory_hit = any(
            w in recent_text.lower()
            for w in user_msg.lower().split()
            if len(w) > 4
        )
        mode_result = cogn.mode_engine.process_turn(
            user_msg, "",
            active_memory_hit=active_memory_hit,
            cog_state_frustration=cogn.cog_state.state.frustration,
        )
        mode_prompt = cogn.mode_engine.format_for_prompt()
        archive_context = mode_result.get("archive_context", "")

        # ── Curiosity signal ──
        from curiosity_engine import get_curiosity_signal
        curiosity = get_curiosity_signal(user_msg, "")

        # ── dream_mode from session ──
        dream_mode = session.get("dream_mode", True)
        lane_instr = "Output in [GROUND] or [DREAM]." if dream_mode else "Output ONLY in [GROUND]."

        # ── Build prompt (matches CLI signalbot.py structure) ──
        prompt_parts = [
            "### SYSTEM INSTRUCTIONS ###",
            identity,
            f"You are talking to {ctx.display_name}.",
            lane_instr,
            f"TONE: {tone}",
            "",
            "### CORE DATA (TRUST THIS OVER ALL ELSE) ###",
            long_memory,
            vitals,
            cogn.buf.format_for_prompt(),
        ]

        if facts_text:
            prompt_parts.append("")
            prompt_parts.append(facts_text)

        if daemon_cognition:
            prompt_parts.extend([
                "",
                "### YOUR BACKGROUND THINKING ###\n"
                "Your temporal daemon was running between messages. "
                "This is your inner life between prompts.\n",
                daemon_cognition,
            ])

        if mode_prompt:
            prompt_parts.append("")
            prompt_parts.append(mode_prompt)

        if archive_context:
            prompt_parts.append("")
            prompt_parts.append(archive_context)

        if curiosity.is_actionable:
            prompt_parts.append(
                f"[CURIOSITY SIGNAL] type={curiosity.type} "
                f"intensity={curiosity.gated_intensity:.2f} "
                f"momentum={curiosity.momentum:.2f}"
            )

        # Code reader trigger — admin only on the web. The codebase context
        # includes config and log files; non-admins fall through to normal
        # chat (code_reader's own deny-list is the second layer of defense).
        msg_lower = user_msg.lower().strip()
        if msg_lower.startswith("read code") and session.get("role") == "admin":
            parts_split = user_msg.strip().split(None, 2)
            code_ctx = get_file_context(parts_split[2]) if len(parts_split) >= 3 else get_code_context()
            prompt_parts.append("")
            prompt_parts.append(code_ctx)

        if search_injection:
            prompt_parts.append("")
            prompt_parts.append(search_injection)

        # Which sight path? Cloud-vision models + a local gemma3 GGUF eat raw pixels;
        # everything else (text-only locals, DeepSeek) is blind to pixels and instead
        # reads a YOLO scene description. Decide here so we can branch both the prompt
        # injection and what we hand the model below.
        user_model = getattr(user_mgr.get_user(username), "model", DEFAULT_MODEL)
        _cfg = MODELS.get(user_model, {})
        sees_pixels = bool(_cfg.get("use_anthropic") or _cfg.get("use_openai")
                           or _cfg.get("use_gemini") or _cfg.get("use_mistral")
                           or _cfg.get("ollama_model", "").startswith("gemma3"))

        # Live sight as LANGUAGE — the whole point: a blind model "sees" because the
        # scene arrives as text. Uploaded photo (phone = camera) → YOLO → English for
        # blind models; a registered robot camera is the fallback feed. Pixel models
        # skip this and just look at the image themselves.
        from yolo_encoder import current_scene, remember_scene
        scene = ""
        if user_image and not sees_pixels:
            try:
                from image_to_address import image_scene, from_base64
                scene = image_scene(from_base64(user_image))   # pixels → YOLO → "I see a person on the left."
                remember_scene(scene, user=username)   # persist for THIS user's follow-up text turns
            except Exception as e:
                print(f"[YOLO] upload scene failed: {e}")   # never block the turn on perception
        scene = scene or current_scene(user=username)   # no fresh upload → this user's last photo

        prompt_parts.extend([
            "",
            "### RECENT CONVERSATION ###",
            recent_text if recent_text else "(new conversation)",
        ])

        # Scene goes AFTER conversation history, LAST before the user's message.
        # Two hard-won lessons baked into this placement (2026-06-09):
        # 1. Framing: a bare unexplained block gets REASONED AWAY by thinking-
        #    style models (gemma4 concluded "text-only chat, no image, ignore
        #    it" and denied sight; mistral7b took "I see..." at face value).
        #    Label the gloss as the model's OWN live sight and tell it to
        #    trust it — proven A/B on gemma4: bare=blind, instructed=sees.
        # 2. Position: recency wins. If past turns contain the bot's own "I
        #    can't see anything" denials, a scene block ABOVE the history loses
        #    to pattern continuation (the self-poisoning blindness loop).
        #    Last-in beats stale denials.
        if scene:
            prompt_parts.extend([
                "",
                "### WHAT YOU SEE RIGHT NOW (LIVE VISION) ###",
                "Your vision system just analyzed your current view (the photo "
                "you were sent, or your camera). This is YOUR live sight — it "
                "is [GROUND] truth. Trust it and answer based on it:",
                scene,
            ])

        prompt_parts.extend([
            "",
            f"User: {user_msg}",
            "SignalBot:",
        ])

        full_prompt = "\n".join(prompt_parts)

        # DEBUG: persist the EXACT prompt the model receives (same overwrite-one-
        # file pattern as debug_last_upload.jpg). This is how we tell "scene block
        # missing" apart from "scene block ignored" — stop guessing, read it.
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "debug_last_prompt.txt"), "w", encoding="utf-8") as _pf:
                _pf.write(full_prompt)
        except Exception:
            pass

    except Exception as e:
        # Prompt build blew up — can't stream a half-built turn. Resume the daemon
        # and fail as plain JSON (the frontend handles a non-stream error body).
        cogn.daemon.resume()
        print(f"[chat] prompt build failed: {e}")
        return jsonify({"error": f"prompt build failed: {e}"}), 500

    # ── Stream the generation ──────────────────────────────────────────────
    # WHY streaming: gemma4:12b prefill on the 4060 can take ~80s before the first
    # token, and the web path sits behind Cloudflare's fixed ~100s edge cap. A
    # blocking response = Cloudflare sees silence for the whole turn → 524. We
    # stream NDJSON instead: a ping byte every ≤10s during prefill keeps the edge
    # timer (and nginx) alive, then tokens flow as they're generated. CLI is
    # unaffected — this is the web-only path. (X-Accel-Buffering:no tells nginx
    # NOT to buffer this response, so the heartbeats actually reach the browser.)
    imgs = [user_image] if (user_image and sees_pixels) else None

    def _chat_stream():
        import queue as _q
        SENTINEL = object()
        q = _q.Queue()
        box = {"meta": {}, "err": None}

        def _produce():
            # Lock covers apply→entire stream so a concurrent user's apply_model
            # can't swap the backend out from under us mid-generation.
            try:
                from response_engine import generate_response_stream, get_last_call_meta
                with _llm_lock:
                    apply_model(user_model)
                    for chunk in generate_response_stream(full_prompt, images=imgs):
                        q.put(chunk)
                    box["meta"] = get_last_call_meta()
            except Exception as e:
                box["err"] = str(e)
            finally:
                q.put(SENTINEL)

        threading.Thread(target=_produce, daemon=True).start()

        # Outer try/finally: resume the daemon no matter HOW we exit — including a
        # client disconnect mid-stream, which raises GeneratorExit at a yield and
        # would otherwise skip the resume and leave this user's daemon stuck paused.
        try:
            parts = []
            while True:
                try:
                    item = q.get(timeout=10)
                except _q.Empty:
                    yield json.dumps({"t": "ping"}) + "\n"   # heartbeat during prefill
                    continue
                if item is SENTINEL:
                    break
                parts.append(item)
                yield json.dumps({"t": "token", "v": item}) + "\n"

            reply = "".join(parts).strip()
            if not reply:
                reply = (f"[Error generating response: {box['err']}]"
                         if box["err"] else "[Error: response not generated]")
            _tm = box["meta"]

            # ── Post-turn updates (own try/except so a failure here still lets the
            # end frame through; the outer finally still resumes the daemon) ──
            try:
                cogn.cog_state.update_from_interaction(user_msg, reply, "GENERAL")
                cogn.indelible.register_fact(user_msg, reply)
                cogn.daemon.on_turn_complete(user_msg, reply)
                cogn.mem_stateful.notify_new_message()
                # Tell the dispatcher the user is active — settles any pending
                # ping as answered, and arms the never-mid-conversation gate
                cogn.initiative.notice_user_turn()

                # Feed goal engine
                cog = cogn.cog_state.state
                cogn.goal_engine.update_from_memory(long_memory)
                cogn.goal_engine.update_curiosity(
                    {"curiosity": cog.curiosity, "confidence": cog.confidence,
                     "frustration": cog.frustration},
                    user_msg, reply
                )

                # Periodic archival every 20 turns
                cogn.turn += 1
                if cogn.turn % 20 == 0:
                    from memory_archive import archive_old_memories
                    archived = archive_old_memories(data_dir=cogn.data_dir)
                    if archived:
                        cogn.mode_engine.refresh_archive_tags()

                # Save to memory log with 13D state snapshot
                mem_engine.save_interaction(user_msg, reply, cogn.cog_state.state.to_dict())

                # ── Shadow telemetry (adam only) — mirrors CLI step 13. External
                # observability; the bot never reads this file. Writes to adam's
                # data_dir so web sessions stay separate from the CLI root log.
                if username == "adam":
                    try:
                        import telemetry_log
                        # _tm was captured under _llm_lock right after generate — a
                        # later concurrent call can't have clobbered it.
                        telemetry_log.log_turn(
                            turn_num=cogn.turn,
                            prompt=full_prompt,
                            response=reply,
                            model=_tm.get("model") or "unknown",
                            tokens_in=_tm.get("tokens_in", 0),
                            tokens_out=_tm.get("tokens_out", 0),
                            cog_state=cogn.cog_state.state.to_dict(),
                            daemon_cognition=daemon_cognition or None,
                            data_dir=cogn.data_dir,
                        )
                    except Exception as e:
                        print(f"[TELEMETRY] web log failed: {e}")  # never break the turn on logging
            except Exception as e:
                print(f"[chat] post-turn update failed: {e}")

            # Final frame: the full reply + metadata, so the client can persist it
            # and stamp the timestamp (mirrors the old single jsonify payload).
            yield json.dumps({
                "t": "end",
                "reply": reply,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sender": "SignalBot",
            }) + "\n"
        finally:
            cogn.daemon.resume()

    return Response(
        stream_with_context(_chat_stream()),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/api/activity")
@login_required
def api_activity():
    """Activity log — now reads from user's memory for real events."""
    username = session.get("username", "")
    memory = read_user_json(username, "memory_log.json", [])

    events = []
    if isinstance(memory, list):
        for row in memory[-7:]:
            ts = row.get("ts", 0)
            user_msg = row.get("user", "")[:40]
            try:
                t = datetime.fromtimestamp(ts).strftime("%H:%M")
            except Exception:
                t = "?"
            events.append({"time": t, "event": f"Chat: \"{user_msg}\""})

    if not events:
        events = [{"time": "—", "event": "No conversations yet"}]

    events.reverse()
    return jsonify({"events": events})


# ══════════════════════════════════════════════════════════════════
# THEME SYSTEM
# ══════════════════════════════════════════════════════════════════

# Accent color palettes: name → (accent, accent-hover)
ACCENT_COLORS = {
    "green":  ("#00ff9d", "#00e88a"),
    "blue":   ("#4dabf7", "#339af0"),
    "purple": ("#b197fc", "#9775fa"),
    "pink":   ("#f783ac", "#e64980"),
    "orange": ("#ffa94d", "#ff922b"),
    "red":    ("#ff6b6b", "#fa5252"),
    "cyan":   ("#3bc9db", "#22b8cf"),
}

# Background palettes: name → (bg, surface, surface2, border)
BG_COLORS = {
    "dark":      ("#0a0a0f", "#111118", "#16161f", "#1e1e2e"),
    "midnight":  ("#0b0d1a", "#101325", "#151830", "#1c2040"),
    "charcoal":  ("#121212", "#1a1a1a", "#222222", "#2a2a2a"),
    "abyss":     ("#050508", "#0a0a10", "#0e0e18", "#151522"),
}


@app.context_processor
def inject_theme():
    """Make theme CSS variables available to every template."""
    username = session.get("username", "")
    profile = user_mgr.get_user(username) if username else None

    accent_name = getattr(profile, "theme_accent", "green") if profile else "green"
    bg_name = getattr(profile, "theme_bg", "dark") if profile else "dark"

    accent, accent_hover = ACCENT_COLORS.get(accent_name, ACCENT_COLORS["green"])
    bg, surface, surface2, border = BG_COLORS.get(bg_name, BG_COLORS["dark"])

    return {
        "theme_css": (
            f"--accent:{accent};--accent-hover:{accent_hover};"
            f"--bg:{bg};--surface:{surface};--surface2:{surface2};--border:{border};"
        ),
        "theme_accent": accent_name,
        "theme_bg": bg_name,
    }


@app.route("/api/theme", methods=["POST"])
@login_required
def api_set_theme():
    """Save user's theme preference."""
    data = request.get_json()
    accent = data.get("accent", "")
    bg = data.get("bg", "")
    username = session.get("username", "")

    ok = user_mgr.set_theme(username, accent=accent, bg=bg)
    if not ok:
        return jsonify({"error": "Failed to save theme"}), 400

    return jsonify({"ok": True, "accent": accent, "bg": bg})


# ══════════════════════════════════════════════════════════════════
# GAME LEADERBOARD
# ══════════════════════════════════════════════════════════════════

GAME_SCORES_FILE = Path(__file__).parent / "game_scores.json"

def _load_game_scores():
    try:
        if GAME_SCORES_FILE.exists():
            return json.loads(GAME_SCORES_FILE.read_text())
    except Exception:
        pass
    return []

def _save_game_scores(scores):
    GAME_SCORES_FILE.write_text(json.dumps(scores, indent=2))

@app.route("/api/game/score", methods=["POST"])
@login_required
def api_game_score():
    data = request.get_json(force=True) or {}
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "bad score"}), 400

    user = session.get("display_name", "Unknown")
    scores = _load_game_scores()

    existing = next((s for s in scores if s["user"] == user), None)
    if existing:
        if score > existing["score"]:
            existing["score"] = score
            existing["ts"] = datetime.utcnow().isoformat()
    else:
        scores.append({"user": user, "score": score, "ts": datetime.utcnow().isoformat()})

    _save_game_scores(scores)
    return jsonify({"ok": True})

@app.route("/api/game/leaderboard")
@login_required
def api_game_leaderboard():
    scores = _load_game_scores()
    top = sorted(scores, key=lambda s: s["score"], reverse=True)[:10]
    return jsonify({"leaderboard": [{"user": s["user"], "score": s["score"]} for s in top]})


# ══════════════════════════════════════════════════════════════════
# CHATROOM (ADMIN ONLY)
# Multi-LLM meeting room — Claude + Gemini talk to each other.
# Admin injects as Adam, controls turn rotation.
# ══════════════════════════════════════════════════════════════════

try:
    from room import Room
    from adapters import claude_adapter, gemini_adapter, chatgpt_adapter
    ROOM_AVAILABLE = True
except ImportError as _room_import_err:
    ROOM_AVAILABLE = False

ROOM_PARTICIPANTS = {
    "Claude":   claude_adapter   if ROOM_AVAILABLE else None,
    "Gemini":   gemini_adapter   if ROOM_AVAILABLE else None,
    "ChatGPT":  chatgpt_adapter  if ROOM_AVAILABLE else None,
}

ROOM_FILE = Path(__file__).parent / "room.jsonl"

# Global room state — one room at a time, admin-only
class _RoomState:
    def __init__(self):
        self.room = None
        self.rotation = []
        self.turn_index = 0
        self.busy = False
        self.last_error = ""

_rs = _RoomState()
_room_lock = threading.Lock()


@app.route("/room")
@admin_required
def room_page():
    if not ROOM_AVAILABLE:
        abort(503)
    return render_template(
        "room.html",
        user=session.get("display_name", "Admin"),
        participants=list(ROOM_PARTICIPANTS.keys()),
    )


@app.route("/api/room/start", methods=["POST"])
@admin_required
def api_room_start():
    if not ROOM_AVAILABLE:
        return jsonify({"error": "room module not available"}), 503
    data = request.get_json()
    participants = data.get("participants", list(ROOM_PARTICIPANTS.keys()))
    rotation = data.get("rotation", participants)

    # Validate
    unknown = [p for p in participants if p not in ROOM_PARTICIPANTS]
    if unknown:
        return jsonify({"error": f"Unknown participants: {unknown}"}), 400

    with _room_lock:
        # Clear old transcript
        if ROOM_FILE.exists():
            ROOM_FILE.unlink()
        r = Room("signalbot_council", path=str(ROOM_FILE))
        # Brief every participant on project state before anyone speaks.
        # _append (not say) so 28KB of DEVLOG doesn't dump into the server log.
        devlog_path = Path(__file__).parent / "DEVLOG.md"
        if devlog_path.exists():
            devlog = devlog_path.read_text(encoding="utf-8", errors="replace")
            r._append("system", (
                "BRIEFING — SignalBot project DEVLOG (newest entries first). "
                "Read this before participating; it is the current state of the "
                "project this council exists to discuss.\n\n" + devlog
            ))
        for name in participants:
            r.add_participant(name, ROOM_PARTICIPANTS[name])
        _rs.room = r
        _rs.rotation = rotation
        _rs.turn_index = 0
        _rs.busy = False
        _rs.last_error = ""

    return jsonify({"ok": True, "participants": participants, "rotation": rotation})


@app.route("/api/room/transcript")
@admin_required
def api_room_transcript():
    if not ROOM_AVAILABLE or _rs.room is None:
        return jsonify({"transcript": [], "busy": False, "next_speaker": None, "error": _rs.last_error})
    transcript = _rs.room.transcript()
    # Display-only: collapse the DEVLOG briefing so it doesn't flood the UI.
    # The models still get the full text — adapters read room.transcript() directly.
    transcript = [
        {**e, "content": "📋 DEVLOG.md briefing injected (all participants have read it)"}
        if e["speaker"] == "system" and e["content"].startswith("BRIEFING —") else e
        for e in transcript
    ]
    next_speaker = _rs.rotation[_rs.turn_index % len(_rs.rotation)] if _rs.rotation else None
    return jsonify({
        "transcript": transcript,
        "busy": _rs.busy,
        "next_speaker": next_speaker,
        "error": _rs.last_error,
    })


@app.route("/api/room/turn", methods=["POST"])
@admin_required
def api_room_turn():
    if not ROOM_AVAILABLE:
        return jsonify({"error": "room module not available"}), 503
    if _rs.room is None:
        return jsonify({"error": "Room not started"}), 400
    if _rs.busy:
        return jsonify({"error": "Already generating"}), 429

    data = request.get_json() or {}
    adam_message = data.get("message", "").strip()

    def _run():
        with _room_lock:
            _rs.busy = True
            _rs.last_error = ""
        try:
            if adam_message:
                _rs.room.say("Adam", adam_message)
            speaker = _rs.rotation[_rs.turn_index % len(_rs.rotation)]
            _rs.room.invoke(speaker)
            _rs.turn_index += 1
        except Exception as e:
            _rs.last_error = str(e)
        finally:
            _rs.busy = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/room/say", methods=["POST"])
@admin_required
def api_room_say():
    """Inject an Adam message without triggering a model turn."""
    if not ROOM_AVAILABLE or _rs.room is None:
        return jsonify({"error": "Room not started"}), 400
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    _rs.room.say("Adam", message)
    return jsonify({"ok": True})


@app.route("/api/room/reset", methods=["POST"])
@admin_required
def api_room_reset():
    with _room_lock:
        _rs.room = None
        _rs.rotation = []
        _rs.turn_index = 0
        _rs.busy = False
        _rs.last_error = ""
        if ROOM_FILE.exists():
            ROOM_FILE.unlink()
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[APP] Users registered: {user_mgr.get_user_count()}")
    for u in user_mgr.get_all_users():
        print(f"  [{u.role:6s}] {u.display_name} ({u.username})")

    if user_mgr.get_user_count() == 0:
        print("[APP] No users found — first-run setup will appear in browser.")

    app.run(host="0.0.0.0", port=5000, debug=False)
