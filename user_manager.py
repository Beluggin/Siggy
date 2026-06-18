# user_manager.py
"""
═══════════════════════════════════════════════════════════════════
USER MANAGER — Multi-User Profile Infrastructure for SignalBot
═══════════════════════════════════════════════════════════════════

PURPOSE:
  Each user gets their own SignalBot experience:
  - Separate memory pipeline (memory_log, cognitive_state, etc.)
  - Separate indelible facts
  - Separate cognitive state vectors
  - Per-user daemon (when active)

  Admin (god mode) can view all users, change permissions, ban.

DIRECTORY STRUCTURE:
  users/
    _global/                  ← shared: identity, ethics
      signal_identity.txt
      signal_ethics.py
    adam/                     ← per-user data directory
      memory_log.json
      cognitive_state.json
      indelible_facts.json
      memory_archive.json
      memory_index.json
      master_summary.json
      behavior_log.json
      plan_buffer.json
    sophie/
      ...

  user_registry.json          ← user accounts, roles, hashes

ROLES:
  "admin"  — god mode. Can view all users, edit permissions, ban.
  "user"   — normal access. Talks to their own SignalBot instance.
  "banned" — locked out. Cannot log in.

SECURITY NOTE:
  Passphrases are hashed with bcrypt (if available) or SHA-256 + salt.
  This is a home network app, not a bank. But we do it properly anyway.
"""

import json
import os
import hashlib
import secrets
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

# Base directory — set this to your SignalBot root
# All user data lives under BASE_DIR/users/
BASE_DIR = Path(__file__).parent
USERS_DIR = BASE_DIR / "users"
GLOBAL_DIR = USERS_DIR / "_global"
REGISTRY_PATH = BASE_DIR / "user_registry.json"

# Files that each user gets their own copy of
USER_DATA_FILES = [
    "memory_log.json",
    "cognitive_state.json",
    "indelible_facts.json",
    "memory_archive.json",
    "memory_index.json",
    "master_summary.json",
    "behavior_log.json",
    "plan_buffer.json",
]

# Files that are shared globally (read from _global/)
GLOBAL_FILES = [
    "signal_identity.txt",
    "signal_ethics.py",
]


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """A registered user account."""
    username: str                  # lowercase, filesystem-safe
    display_name: str              # what they see on screen
    role: str = "user"             # "admin", "user", "banned"
    passphrase_hash: str = ""      # hashed passphrase
    salt: str = ""                 # for SHA-256 fallback
    created_ts: float = 0.0
    last_login_ts: float = 0.0
    login_count: int = 0
    # Admin notes (only visible in god mode)
    admin_notes: str = ""
    # Model preference — set by admin per user
    model: str = "mistral_med"
    # Theme preferences
    theme_accent: str = "green"
    theme_bg: str = "dark"
    # Initiative pings (daemon-initiated chat messages) — opt-in per user
    initiative_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'UserProfile':
        # Filter to only known fields
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_banned(self) -> bool:
        return self.role == "banned"

    @property
    def data_dir(self) -> Path:
        """Path to this user's data directory."""
        return USERS_DIR / self.username


# ═══════════════════════════════════════════════════════════════════
# PASSWORD HASHING
# ═══════════════════════════════════════════════════════════════════

def _hash_passphrase(passphrase: str, salt: str = "") -> tuple:
    """
    Hash a passphrase. Returns (hash, salt).
    Uses SHA-256 + salt. Good enough for a home network app.
    """
    if not salt:
        salt = secrets.token_hex(16)
    combined = f"{salt}:{passphrase}"
    hashed = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return hashed, salt


def _verify_passphrase(passphrase: str, stored_hash: str, salt: str) -> bool:
    """Check if a passphrase matches the stored hash."""
    computed, _ = _hash_passphrase(passphrase, salt)
    return secrets.compare_digest(computed, stored_hash)


# ═══════════════════════════════════════════════════════════════════
# USER MANAGER
# ═══════════════════════════════════════════════════════════════════

class UserManager:
    """
    Manages user accounts, authentication, and data directories.

    Thread-safe for reads. Writes (create/update/delete) should
    only happen from the main thread or Flask request handlers.
    """

    def __init__(self):
        self._users: Dict[str, UserProfile] = {}
        self._ensure_directories()
        self._load_registry()

    # ═══ SETUP ═══

    def _ensure_directories(self):
        """Create the directory structure if it doesn't exist."""
        USERS_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

        # Copy global files to _global/ if they don't exist there yet
        for filename in GLOBAL_FILES:
            src = BASE_DIR / filename
            dst = GLOBAL_DIR / filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"[USERS] Copied {filename} to _global/")

    def _load_registry(self):
        """Load user registry from disk."""
        if not REGISTRY_PATH.exists():
            self._users = {}
            return

        try:
            data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            self._users = {}
            for username, user_data in data.items():
                user_data["username"] = username
                self._users[username] = UserProfile.from_dict(user_data)
        except Exception as e:
            print(f"[USERS] Failed to load registry: {e}")
            self._users = {}

    def _save_registry(self):
        """Save user registry to disk."""
        data = {}
        for username, profile in self._users.items():
            d = profile.to_dict()
            # Don't store username redundantly as key AND field
            del d["username"]
            data[username] = d

        REGISTRY_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # ═══ USER CREATION ═══

    def _sanitize_username(self, name: str) -> str:
        """
        Convert display name to filesystem-safe username.
        Lowercase, alphanumeric + underscore only.
        """
        safe = "".join(
            c if c.isalnum() else "_"
            for c in name.lower().strip()
        )
        # Remove leading/trailing underscores and collapse doubles
        while "__" in safe:
            safe = safe.replace("__", "_")
        safe = safe.strip("_")
        return safe or "user"

    def create_user(
        self,
        display_name: str,
        passphrase: str,
        role: str = "user",
        admin_notes: str = "",
    ) -> Optional[UserProfile]:
        """
        Create a new user account and data directory.

        Returns the UserProfile on success, None if username taken.
        """
        username = self._sanitize_username(display_name)

        # Check for duplicates
        if username in self._users:
            print(f"[USERS] Username '{username}' already exists")
            return None

        # Reserved names
        if username in ("_global", "admin", "system", "signalbot"):
            print(f"[USERS] Username '{username}' is reserved")
            return None

        # Hash passphrase
        hashed, salt = _hash_passphrase(passphrase)

        # Create profile
        profile = UserProfile(
            username=username,
            display_name=display_name,
            role=role,
            passphrase_hash=hashed,
            salt=salt,
            created_ts=time.time(),
            admin_notes=admin_notes,
        )

        # Create data directory with empty starter files
        self._create_user_directory(profile)

        # Register
        self._users[username] = profile
        self._save_registry()

        print(f"[USERS] Created user '{display_name}' ({username}) as {role}")
        return profile

    def _create_user_directory(self, profile: UserProfile):
        """
        Create the per-user data directory with empty starter files.
        Each file starts as an empty JSON structure so SignalBot
        doesn't crash on first load.
        """
        user_dir = profile.data_dir
        user_dir.mkdir(parents=True, exist_ok=True)

        # Default empty content for each file type
        defaults = {
            "memory_log.json": [],
            "cognitive_state.json": {
                "frustration": 0.2,
                "curiosity": 0.8,
                "confidence": 0.6,
                "engagement": 0.9,
                "identity_adherence": 0.7,
                "context": 0.9,
                "tone": {
                    "playful": 0.6,
                    "formal": 0.3,
                    "concise": 0.5,
                    "warm": 0.7,
                },
                "cognitive_load": 0.7,
                "recursion_tolerance": 0.5,
                "affect_matching": 0.6,
                "last_update": time.time(),
            },
            "indelible_facts.json": {"facts": [], "last_updated": 0},
            "memory_archive.json": [],
            "memory_index.json": {"items": []},
            "master_summary.json": {
                "facts": [],
                "active_projects": [],
                "preferences": [],
            },
            "behavior_log.json": {"events": []},
            "plan_buffer.json": {"plans": {}, "archive": []},
        }

        for filename in USER_DATA_FILES:
            filepath = user_dir / filename
            if not filepath.exists():
                content = defaults.get(filename, {})
                filepath.write_text(
                    json.dumps(content, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        print(f"[USERS] Created data directory: {user_dir}")

    # ═══ AUTHENTICATION ═══

    def authenticate(self, username_or_display: str, passphrase: str) -> Optional[UserProfile]:
        """
        Authenticate a user by username or display name + passphrase.
        Returns the UserProfile on success, None on failure.
        """
        # Try exact username match first
        profile = self._users.get(username_or_display.lower().strip())

        # Try display name match
        if not profile:
            sanitized = self._sanitize_username(username_or_display)
            profile = self._users.get(sanitized)

        if not profile:
            return None

        if profile.is_banned:
            print(f"[USERS] Banned user '{profile.username}' attempted login")
            return None

        if not _verify_passphrase(passphrase, profile.passphrase_hash, profile.salt):
            return None

        # Update login stats
        profile.last_login_ts = time.time()
        profile.login_count += 1
        self._save_registry()

        return profile

    # ═══ USER QUERIES ═══

    def get_user(self, username: str) -> Optional[UserProfile]:
        """Get a user profile by username."""
        return self._users.get(username)

    def get_all_users(self) -> List[UserProfile]:
        """Get all user profiles. For admin panel."""
        return list(self._users.values())

    def user_exists(self, username: str) -> bool:
        return username in self._users

    def get_user_count(self) -> int:
        return len(self._users)

    # ═══ ADMIN ACTIONS (GOD MODE) ═══

    def set_role(self, username: str, new_role: str) -> bool:
        """
        Change a user's role. Admin only.
        Valid roles: "admin", "user", "banned"
        """
        if new_role not in ("admin", "user", "banned"):
            return False

        profile = self._users.get(username)
        if not profile:
            return False

        old_role = profile.role
        profile.role = new_role
        self._save_registry()
        print(f"[USERS] {username}: {old_role} → {new_role}")
        return True

    def set_admin_notes(self, username: str, notes: str) -> bool:
        """Set admin notes on a user profile."""
        profile = self._users.get(username)
        if not profile:
            return False
        profile.admin_notes = notes
        self._save_registry()
        return True

    def change_passphrase(self, username: str, new_passphrase: str) -> bool:
        """Reset a user's passphrase. Admin action."""
        profile = self._users.get(username)
        if not profile:
            return False

        hashed, salt = _hash_passphrase(new_passphrase)
        profile.passphrase_hash = hashed
        profile.salt = salt
        self._save_registry()
        print(f"[USERS] Passphrase reset for {username}")
        return True

    def set_model(self, username: str, model: str) -> bool:
        """Set the LLM model for a specific user. Admin action."""
        profile = self._users.get(username)
        if not profile:
            return False
        profile.model = model
        self._save_registry()
        print(f"[USERS] Model for {username} set to {model}")
        return True

    def set_theme(self, username: str, accent: str = "", bg: str = "") -> bool:
        """Save theme preferences for a user."""
        profile = self._users.get(username)
        if not profile:
            return False
        if accent:
            profile.theme_accent = accent
        if bg:
            profile.theme_bg = bg
        self._save_registry()
        return True

    def set_initiative(self, username: str, enabled: bool) -> bool:
        """Toggle daemon-initiated pings for a user."""
        profile = self._users.get(username)
        if not profile:
            return False
        profile.initiative_enabled = bool(enabled)
        self._save_registry()
        print(f"[USERS] Initiative for {username} set to {enabled}")
        return True

    def delete_user(self, username: str, delete_data: bool = False) -> bool:
        """
        Remove a user account.
        If delete_data=True, also removes their data directory.
        If delete_data=False, data is preserved (can be reassigned).
        """
        profile = self._users.get(username)
        if not profile:
            return False

        if delete_data and profile.data_dir.exists():
            shutil.rmtree(profile.data_dir)
            print(f"[USERS] Deleted data directory for {username}")

        del self._users[username]
        self._save_registry()
        print(f"[USERS] Deleted user: {username}")
        return True

    # ═══ USER DATA ACCESS ═══

    def get_user_data_path(self, username: str) -> Optional[Path]:
        """
        Get the path to a user's data directory.
        Returns None if user doesn't exist.
        """
        profile = self._users.get(username)
        if not profile:
            return None
        return profile.data_dir

    def get_user_file(self, username: str, filename: str) -> Optional[Path]:
        """
        Get the path to a specific file in a user's data directory.
        Returns None if user doesn't exist or file isn't allowed.
        """
        if filename not in USER_DATA_FILES:
            return None
        profile = self._users.get(username)
        if not profile:
            return None
        return profile.data_dir / filename

    def read_user_data(self, username: str, filename: str) -> Optional[Any]:
        """
        Read and parse a JSON file from a user's data directory.
        For admin viewing.
        """
        filepath = self.get_user_file(username, filename)
        if not filepath or not filepath.exists():
            return None
        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_user_stats(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a user's data for the admin panel.
        """
        profile = self._users.get(username)
        if not profile:
            return None

        stats = {
            "username": profile.username,
            "display_name": profile.display_name,
            "role": profile.role,
            "created": profile.created_ts,
            "last_login": profile.last_login_ts,
            "login_count": profile.login_count,
            "admin_notes": profile.admin_notes,
        }

        # Count memories
        mem_path = profile.data_dir / "memory_log.json"
        if mem_path.exists():
            try:
                mem = json.loads(mem_path.read_text(encoding="utf-8"))
                stats["memory_count"] = len(mem) if isinstance(mem, list) else 0
            except Exception:
                stats["memory_count"] = 0
        else:
            stats["memory_count"] = 0

        # Count archived episodes + store recent 4 for popup
        archive_path = profile.data_dir / "memory_archive.json"
        if archive_path.exists():
            try:
                archive = json.loads(archive_path.read_text(encoding="utf-8"))
                if isinstance(archive, list):
                    stats["archive_episodes"] = len(archive)
                    stats["archive_recent"] = sorted(
                        archive, key=lambda e: e.get("ts_end", 0), reverse=True
                    )[:4]
                else:
                    stats["archive_episodes"] = 0
                    stats["archive_recent"] = []
            except Exception:
                stats["archive_episodes"] = 0
                stats["archive_recent"] = []
        else:
            stats["archive_episodes"] = 0
            stats["archive_recent"] = []

        # Count indelible facts + store list for popup
        facts_path = profile.data_dir / "indelible_facts.json"
        if facts_path.exists():
            try:
                facts_data = json.loads(facts_path.read_text(encoding="utf-8"))
                raw_facts = facts_data.get("facts", [])
                stats["indelible_facts"] = len(raw_facts)
                stats["facts_list"] = raw_facts
            except Exception:
                stats["indelible_facts"] = 0
                stats["facts_list"] = []
        else:
            stats["indelible_facts"] = 0
            stats["facts_list"] = []

        # Cognitive state snapshot — full numeric vector for popup
        cog_path = profile.data_dir / "cognitive_state.json"
        if cog_path.exists():
            try:
                cog = json.loads(cog_path.read_text(encoding="utf-8"))
                stats["cognitive_state"] = {
                    "curiosity": cog.get("curiosity", 0),
                    "frustration": cog.get("frustration", 0),
                    "engagement": cog.get("engagement", 0),
                    "confidence": cog.get("confidence", 0),
                }
                stats["cognitive_state_full"] = {
                    k: v for k, v in cog.items() if isinstance(v, (int, float))
                }
            except Exception:
                stats["cognitive_state"] = None
                stats["cognitive_state_full"] = {}
        else:
            stats["cognitive_state"] = None
            stats["cognitive_state_full"] = {}

        return stats

    # ═══ DATA MIGRATION ═══

    def migrate_existing_data(self, username: str) -> bool:
        """
        Move existing SignalBot data files from the root directory
        into a user's namespace. For migrating your current data
        into your admin account.

        Only copies files that exist in the root AND are in
        USER_DATA_FILES. Won't overwrite if user already has data.
        """
        profile = self._users.get(username)
        if not profile:
            return False

        migrated = 0
        for filename in USER_DATA_FILES:
            src = BASE_DIR / filename
            dst = profile.data_dir / filename
            if src.exists():
                if dst.exists():
                    # Check if dst is just the empty default
                    try:
                        dst_data = json.loads(dst.read_text(encoding="utf-8"))
                        is_empty = (
                            (isinstance(dst_data, list) and len(dst_data) == 0) or
                            (isinstance(dst_data, dict) and
                             not any(dst_data.get(k) for k in dst_data
                                     if k != "last_updated"))
                        )
                    except Exception:
                        is_empty = True

                    if not is_empty:
                        print(f"[MIGRATE] Skipping {filename} — user already has data")
                        continue

                shutil.copy2(src, dst)
                migrated += 1
                print(f"[MIGRATE] Copied {filename} → {username}/")

        if migrated > 0:
            print(f"[MIGRATE] Moved {migrated} files into {username}'s namespace")

        return migrated > 0

    # ═══ FIRST RUN SETUP ═══

    def setup_first_admin(self, display_name: str, passphrase: str) -> UserProfile:
        """
        Create the first admin user. Called on first run.
        Also migrates existing data into this account.
        """
        profile = self.create_user(
            display_name=display_name,
            passphrase=passphrase,
            role="admin",
            admin_notes="System creator. God mode.",
        )

        if profile:
            # Migrate existing SignalBot data into admin account
            self.migrate_existing_data(profile.username)

        return profile


# ═══════════════════════════════════════════════════════════════════
# CONTEXT MANAGER — Per-Request User Context
# ═══════════════════════════════════════════════════════════════════

class UserContext:
    """
    Sets up the file paths for a specific user's session.

    When a user is chatting, SignalBot's modules need to read/write
    from THEIR data directory, not the global one. This context
    object provides the path overrides.

    Usage in Flask:
        ctx = UserContext("sophie", user_manager)
        memory_log_path = ctx.get_path("memory_log.json")
        # → users/sophie/memory_log.json
    """

    def __init__(self, username: str, manager: UserManager):
        self.username = username
        self.profile = manager.get_user(username)
        self._manager = manager

        if not self.profile:
            raise ValueError(f"User '{username}' not found")

    def get_path(self, filename: str) -> Path:
        """
        Get the correct path for a data file.
        User-specific files → user's directory.
        Global files → _global/ directory.
        """
        if filename in GLOBAL_FILES:
            return GLOBAL_DIR / filename
        if filename in USER_DATA_FILES:
            return self.profile.data_dir / filename
        # Unknown file — default to user directory
        return self.profile.data_dir / filename

    @property
    def data_dir(self) -> Path:
        return self.profile.data_dir

    @property
    def is_admin(self) -> bool:
        return self.profile.is_admin

    @property
    def display_name(self) -> str:
        return self.profile.display_name


# ═══════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════

_manager: Optional[UserManager] = None

def get_user_manager() -> UserManager:
    global _manager
    if _manager is None:
        _manager = UserManager()
    return _manager
