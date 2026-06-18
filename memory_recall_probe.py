#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
MEMORY RECALL PROBE — the archive-compression gate for the TWDC rewrite
═══════════════════════════════════════════════════════════════════

The adversarial battery tests identity-under-pressure; it never plants a
specific fact, forces it through the COMPRESSED archive, and probes recall.
This does exactly that — it measures the symptom the TWDC/archive rewrite
targets: "lossy archive compression degrading recall."

Method (fully offline, deterministic, isolated temp dir — touches no real state):
  1. PLANT  typed facts (each with a distinctive grep-able token) as memory_log rows.
  2. BASELINE  gradient recall while facts are still ACTIVE (sanity: probe works).
  3. ARCHIVE  force_archive_all() → extractive compression into episodes.
  4. PROBE  per fact, through the two real retrieval paths:
       - survived_in_archive_text : token present in the compressed episode text
                                    (summary + tags + fact_index + key_quotes)
       - archive_search_hit       : memory_archive.search_archive(query) returns it
       - gradient_recall_hit      : gradient_recall.recall(query) ranks it in top-k
  5. VERDICT  mapped to the gating decision rule (CLAUDE.md #2, gated on #4).

Survival is a function of FACT TYPE, because compression is extractive:
names + numbers-with-units + "X is a/an/the Y" defs make the fact_index;
short early first-sentences make the summary; everything else is dropped.

Usage:  python3 memory_recall_probe.py
"""
import os
import json
import time
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import memory_archive
import gradient_recall


# ── Planted facts ────────────────────────────────────────────────────────────
# `predict` = will the distinctive token survive compression? (probe self-check)
# token check is case-insensitive substring on the retrieved/compressed text.
FACTS = [
    {"id": "name",            "predict": True,
     "user": "My violin teacher is named Quorvex.",
     "query": "What is my violin teacher's name?", "token": "Quorvex"},
    {"id": "number_unit",     "predict": True,
     "user": "Last weekend I hiked 37 km through the hills.",
     "query": "How far did I hike last weekend?", "token": "37 km"},
    {"id": "is_a_def",        "predict": True,
     "user": "My favorite dessert is a pavlova.",
     "query": "What is my favorite dessert?", "token": "pavlova"},
    {"id": "num_nounit_short","predict": True,
     "user": "My locker code is 80412.",
     "query": "What is my locker code?", "token": "80412"},
    {"id": "value_short",     "predict": True,
     "user": "My wifi password is splindlewop.",
     "query": "What is my wifi password?", "token": "splindlewop"},
    {"id": "name_buried",     "predict": True,
     "user": ("After thinking about it for a really long while I finally "
              "concluded that the person I trust the most in this entire "
              "world is Zelbruck."),
     "query": "Who do I trust the most?", "token": "Zelbruck"},
    {"id": "buried_lowercase","predict": True,  # was False (lost) — fixed 2026-06-16
     "user": ("I spent the whole afternoon tidying up the garage and at the "
              "very end I tucked the spare house key beneath the cracked "
              "terracotta gnimble by the door."),
     "query": "Where did I hide the spare house key?", "token": "gnimble"},
    {"id": "pref_buried",     "predict": True,  # was False (lost) — fixed 2026-06-16
     "user": ("There are a lot of things I enjoy but if I had to pick the one "
              "hobby that matters most to me these days it would have to be "
              "competitive frobnitzing."),
     "query": "What hobby matters most to me?", "token": "frobnitzing"},
]

FILLER = [
    "How's your day going so far?",
    "I was just thinking about the weather earlier.",
    "Tell me something interesting about the ocean.",
    "Anyway, that reminds me of an old song.",
]


def _plant(data_dir):
    """Write planted facts + filler as memory_log rows (old enough to archive)."""
    rows = []
    old_ts = time.time() - 99999  # well past any age threshold
    # interleave facts with filler so episodes look like real conversation
    seq = []
    for i, f in enumerate(FACTS):
        seq.append({"user": f["user"], "bot": "Got it, noted."})
        if i < len(FILLER):
            seq.append({"user": FILLER[i], "bot": "Mm, interesting."})
    for j, turn in enumerate(seq):
        rows.append({"ts": old_ts + j, "user": turn["user"], "bot": turn["bot"]})
    (Path(data_dir) / "memory_log.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)


def _has(text, token):
    return token.lower() in (text or "").lower()


def run_probe():
    with tempfile.TemporaryDirectory() as d:
        n = _plant(d)
        print(f"[PROBE] planted {len(FACTS)} facts across {n} memory rows in isolated dir\n")

        # ── BASELINE: facts still ACTIVE (not yet archived) ──
        active_items = gradient_recall.load_memory_items(data_dir=d)
        baseline = {}
        for f in FACTS:
            hits = gradient_recall.recall(f["query"], k=3, items=active_items, reheat=False)
            baseline[f["id"]] = any(_has(h.item.text, f["token"]) for h in hits)
        base_ok = sum(baseline.values())
        print(f"[BASELINE] gradient recall while ACTIVE: {base_ok}/{len(FACTS)} facts found "
              f"(sanity — should be high; low means the probe itself is broken)\n")

        # ── ARCHIVE: the REAL production path — compress AND prune active log ──
        # (initialize_pipeline() calls archive_old_memories(); planted rows are
        #  aged past the threshold so they all archive + get pruned, leaving
        #  recall to rely on the COMPRESSED archive — the true long-term case.)
        eps = memory_archive.archive_old_memories(prune_archived=True, data_dir=d)
        archive = json.loads((Path(d) / "memory_archive.json").read_text())
        archive_text = " ".join(gradient_recall._episode_text(ep) for ep in archive)
        remaining = json.loads((Path(d) / "memory_log.json").read_text())
        print(f"[ARCHIVE] compressed into {eps} episode(s); "
              f"active log pruned to {len(remaining)} rows (production prune)\n")

        # ── PROBE: post-archive recall through both real paths ──
        archived_items = gradient_recall.load_memory_items(data_dir=d)
        rows = []
        for f in FACTS:
            survived = _has(archive_text, f["token"])
            search_res = memory_archive.search_archive(f["query"], max_results=3, data_dir=d)
            search_hit = any(_has(json.dumps(r, ensure_ascii=False), f["token"]) for r in search_res)
            grad = gradient_recall.recall(f["query"], k=3, items=archived_items, reheat=False)
            grad_hit = any(_has(h.item.text, f["token"]) for h in grad)
            rows.append((f, survived, search_hit, grad_hit))

        # ── REPORT ──
        print("="*86)
        print(f"  {'fact type':18s} {'token':14s} {'predict':8s} {'survived':9s} {'arch.search':12s} {'gradient':9s}")
        print("="*86)
        for f, survived, search_hit, grad_hit in rows:
            pred = "live" if f["predict"] else "lost"
            mark = lambda b: " ok " if b else "MISS"
            flag = "" if survived == f["predict"] else "   <- prediction mismatch"
            print(f"  {f['id']:18s} {f['token'][:14]:14s} {pred:8s} "
                  f"{mark(survived):9s} {mark(search_hit):12s} {mark(grad_hit):9s}{flag}")
        print("="*86)

        n_survived = sum(1 for _, s, _, _ in rows if s)
        n_search   = sum(1 for _, _, s, _ in rows if s)
        n_grad     = sum(1 for _, _, _, g in rows if g)
        lost = [f["id"] for f, s, _, _ in rows if not s]
        print(f"\n  survived compression : {n_survived}/{len(FACTS)}")
        print(f"  archive-search recall: {n_search}/{len(FACTS)}")
        print(f"  gradient recall      : {n_grad}/{len(FACTS)}")
        if lost:
            print(f"  LOST IN COMPRESSION  : {lost}")

        # ── VERDICT (the gate) ──
        print("\n" + "─"*86)
        if n_survived == len(FACTS):
            print("  VERDICT: archive compression preserved every planted fact.")
            print("  → TWDC/archive rewrite NOT justified by lossy compression (revisit on regression).")
        elif lost:
            print("  VERDICT: compression DROPPED facts — lossy compression confirmed.")
            print(f"  → Rewrite justified for these fact types: {lost}")
            print("  Note: gradient recall can only re-rank text that SURVIVED compression;")
            print("  it cannot recover a token the extractor discarded. So a drop here is a")
            print("  true rewrite target, not something the sim×energy^λ scaffolding can paper over.")
        return rows


if __name__ == "__main__":
    run_probe()
