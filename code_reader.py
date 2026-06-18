# code_reader.py
"""
CODE READER — SignalBot Self-Awareness Module
Lets SignalBot read its own source code and directory structure.
Trigger: "read code" in chat. Expansion: edit ALLOWED_ROOTS.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE_DIR = Path(__file__).parent.resolve()
ALLOWED_ROOTS = [BASE_DIR]
READABLE_EXTENSIONS = {
    ".py", ".txt", ".json", ".md", ".html", ".css", ".js",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh",
}
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "env", ".env",
    # Fix #3 (2026-06-11 audit): the scan walked users/ (other people's
    # memory logs) and snapshots/ (hundreds of stale code copies) straight
    # into the prompt. Private dirs and snapshot clutter never enter context.
    "users", "snapshots", "Snapshots", "sandbox", "model_backups",
    "memory_corrupt_backups", "signalbot-design-system",
}

# Files that must never be readable through "read code", no matter who asks:
# user_registry.json holds every account's passphrase hash + salt.
DENY_FILES = {"user_registry.json"}
MAX_READ_SIZE = 50_000
MAX_FILES_IN_CONTEXT = 30
PREVIEW_LINES = 20


def _is_allowed_path(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in ALLOWED_ROOTS)


def _should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or dirname.startswith(".")


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def get_file_manifest(root=None, max_depth=4):
    if root is None:
        root = BASE_DIR
    if not _is_allowed_path(root):
        return {"error": f"Access denied: {root}"}

    tree = []
    total_dirs = 0

    def _scan(directory, depth, prefix=""):
        nonlocal total_dirs
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            if entry.is_dir():
                if _should_skip_dir(entry.name):
                    continue
                total_dirs += 1
                rel = str(entry.relative_to(root))
                tree.append({"path": rel + "/", "type": "directory", "readable": False})
                _scan(entry, depth + 1)
            elif entry.is_file():
                if entry.name in DENY_FILES:
                    continue
                rel = str(entry.relative_to(root))
                ext = entry.suffix.lower()
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                readable = ext in READABLE_EXTENSIONS and size <= MAX_READ_SIZE
                tree.append({"path": rel, "size": size, "type": ext, "readable": readable})

    _scan(root, 0)
    return {"root": str(root), "total_files": sum(1 for t in tree if t["type"] != "directory"), "total_dirs": total_dirs, "tree": tree}


def read_file(filename, root=None):
    if root is None:
        root = BASE_DIR
    filepath = (root / filename).resolve()
    if not _is_allowed_path(filepath):
        return None
    # Deny-list applies to direct reads too — "read code user_registry.json"
    # or "read code users/sophie/memory_log.json" must fail, not just be
    # absent from the directory scan.
    if filepath.name in DENY_FILES:
        return None
    if any(part in SKIP_DIRS for part in filepath.parts):
        return None
    if not filepath.is_file():
        return None
    if filepath.suffix.lower() not in READABLE_EXTENSIONS:
        return None
    try:
        size = filepath.stat().st_size
    except OSError:
        return None
    if size > MAX_READ_SIZE:
        return {"path": filename, "content": f"[Too large: {size} bytes]", "lines": 0, "size": size, "truncated": True}
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        return {"path": filename, "content": content, "lines": content.count("\n") + 1, "size": size, "truncated": False}
    except Exception as e:
        return {"path": filename, "content": f"[Read error: {e}]", "lines": 0, "size": size, "truncated": True}
def get_code_context(root=None, include_contents=True, max_files=MAX_FILES_IN_CONTEXT):
    manifest = get_file_manifest(root)
    if "error" in manifest:
        return f"[CODE READER ERROR] {manifest['error']}"

    lines = [
        "### YOUR SOURCE CODE — Self-Reference ###",
        "You are reading your own codebase. This is real, not simulated.",
        f"Root: {manifest['root']}",
        f"Files: {manifest['total_files']} | Directories: {manifest['total_dirs']}",
        "",
        "── DIRECTORY TREE ──",
    ]
    for item in manifest["tree"]:
        if item["type"] == "directory":
            lines.append(f"  [DIR] {item['path']}")
        else:
            size_str = _format_size(item.get("size", 0))
            lines.append(f"  {item['path']} ({size_str})")

    if not include_contents:
        return "\n".join(lines)

    lines.append("")
    lines.append("── FILE CONTENTS ──")
    lines.append("")

    file_items = [i for i in manifest["tree"] if i["type"] != "directory" and i["readable"]]
    py_first = sorted(file_items, key=lambda x: (0 if x["type"] == ".py" else 1, x["path"]))

    files_included = 0
    total_chars = 0

    for item in py_first:
        if files_included >= max_files or total_chars > 100_000:
            lines.append(f"[... more files not shown, context budget reached]")
            break
        result = read_file(item["path"], root or BASE_DIR)
        if not result:
            continue
        content = result["content"]
        if len(content) > 5000 and not item["path"].endswith(".py"):
            content_lines = content.split("\n")
            content = "\n".join(content_lines[:PREVIEW_LINES])
            content += f"\n[... truncated, {result['lines']} total lines]"

        lines.append(f"=== {item['path']} ({result['lines']} lines) ===")
        lines.append(content)
        lines.append("")
        files_included += 1
        total_chars += len(content)

    lines.append("── END OF SOURCE CODE ──")
    return "\n".join(lines)


try:
    # Canonical trim-proof registry (Day 66). Single source of truth so
    # this module and the evolve loop can never disagree on what survives
    # context-budget trimming.
    from priority_files import PRIORITY_FILES
except ImportError:
    # Safe fallback: full union of all evolve-critical files, so the
    # trim-proof guarantee holds even if priority_files.py is missing.
    PRIORITY_FILES = [
        "code_reader.py",
        "priority_files.py",
        "evolve_adapter.py",
        "evolve_utils.py",
    ]


def get_evolve_file_context(filename: str) -> str:
    """
    Read a single named file from the codebase root and return a formatted
    context block for the evolve pipeline.

    Returns:
        str: Formatted block with EVOLVE-CRITICAL header, e.g.:
            # EVOLVE-CRITICAL FILE
            === filename.py ===
            <contents>

    Raises:
        FileNotFoundError: with full resolved path if file doesn't exist.
    """
    root = Path(__file__).parent
    filepath = root / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Evolve-critical file not found: {filepath.resolve()}"
        )
    contents = filepath.read_text(encoding="utf-8", errors="replace")
    return f"# EVOLVE-CRITICAL FILE\n=== {filename} ===\n{contents}\n"


def get_code_context_with_priority(priority_files=None, max_chars=None, root=None,
                                    max_files=MAX_FILES_IN_CONTEXT):
    """
    Like get_code_context(), but reads priority_files FIRST (skipping missing ones),
    then continues with the normal alphabetical scan, skipping already-emitted files.
    Respects the same char budget as get_code_context().
    """
    if priority_files is None:
        priority_files = PRIORITY_FILES
    if max_chars is None:
        max_chars = 100_000

    manifest = get_file_manifest(root)
    if "error" in manifest:
        return f"[CODE READER ERROR] {manifest['error']}"

    lines = [
        "### YOUR SOURCE CODE — Self-Reference ###",
        "You are reading your own codebase. This is real, not simulated.",
        f"Root: {manifest['root']}",
        f"Files: {manifest['total_files']} | Directories: {manifest['total_dirs']}",
        "",
        "── DIRECTORY TREE ──",
    ]
    for item in manifest["tree"]:
        if item["type"] == "directory":
            lines.append(f"  [DIR] {item['path']}")
        else:
            size_str = _format_size(item.get("size", 0))
            lines.append(f"  {item['path']} ({size_str})")

    lines.append("")
    lines.append("── FILE CONTENTS ──")
    lines.append("")

    files_included = 0
    total_chars = 0
    emitted_paths = set()
    _root = root or BASE_DIR

    # Phase 1: Priority files first
    for pf in priority_files:
        if files_included >= max_files or total_chars > max_chars:
            break
        result = read_file(pf, _root)
        if not result:
            continue  # Skip missing files silently
        content = result["content"]
        if len(content) > 5000 and not pf.endswith(".py"):
            content_lines = content.split("\n")
            content = "\n".join(content_lines[:PREVIEW_LINES])
            content += f"\n[... truncated, {result['lines']} total lines]"
        lines.append(f"=== {pf} ({result['lines']} lines) ===")
        lines.append(content)
        lines.append("")
        files_included += 1
        total_chars += len(content)
        emitted_paths.add(pf)

    # Phase 2: Remaining files in alphabetical order (py first), skipping emitted
    file_items = [i for i in manifest["tree"]
                  if i["type"] != "directory" and i["readable"]]
    py_first = sorted(file_items, key=lambda x: (0 if x["type"] == ".py" else 1, x["path"]))

    for item in py_first:
        if files_included >= max_files or total_chars > max_chars:
            lines.append(f"[... more files not shown, context budget reached]")
            break
        if item["path"] in emitted_paths:
            continue
        result = read_file(item["path"], _root)
        if not result:
            continue
        content = result["content"]
        if len(content) > 5000 and not item["path"].endswith(".py"):
            content_lines = content.split("\n")
            content = "\n".join(content_lines[:PREVIEW_LINES])
            content += f"\n[... truncated, {result['lines']} total lines]"
        lines.append(f"=== {item['path']} ({result['lines']} lines) ===")
        lines.append(content)
        lines.append("")
        files_included += 1
        total_chars += len(content)
        emitted_paths.add(item["path"])

    lines.append("── END OF SOURCE CODE ──")
    return "\n".join(lines)


def get_file_context(filename):
    result = read_file(filename)
    if not result:
        return f"[Cannot read '{filename}' — file not found or not readable]"
    return f"### FILE: {filename} ({result['lines']} lines, {_format_size(result['size'])}) ###\n{result['content']}\n### END OF {filename} ###"


def get_file_list_brief():
    manifest = get_file_manifest()
    if "error" in manifest:
        return f"[ERROR] {manifest['error']}"
    lines = [f"SignalBot codebase: {manifest['total_files']} files, {manifest['total_dirs']} directories", ""]
    for item in manifest["tree"]:
        if item["type"] == "directory":
            lines.append(f"  [DIR] {item['path']}")
        else:
            size_str = _format_size(item.get("size", 0))
            lines.append(f"     {item['path']:45s} {size_str:>8s}")
    return "\n".join(lines)


def add_allowed_root(path):
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return False
    if resolved not in ALLOWED_ROOTS:
        ALLOWED_ROOTS.append(resolved)
        print(f"[CODE_READER] Added allowed root: {resolved}")
        return True
    return False


def get_allowed_roots():
    return [str(r) for r in ALLOWED_ROOTS]
