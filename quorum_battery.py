#!/usr/bin/env python3
"""
quorum_battery.py — find SignalBot's voice by cross-model vote.

Every reachable model in app.MODELS runs the SAME N-turn voice battery from a
FIXED identity seed, each model threading its OWN replies (the
initiative_swap_test methodology, one altitude up — a multi-turn conversation
instead of a single ping). No app state is mutated: identity is held byte-fixed
across models, so the only thing that varies is the substrate. That's what makes
the swap valid.

Then the QUORUM votes. For each turn, every reachable model scores every
*anonymized* response 0-10 on voice-fidelity and crowns the single most-Signal
one with a one-line why. Aggregate the ballots → Signal's voice by cross-
substrate consensus, plus the jurors' own words for WHAT that voice is.

On-thesis (CLAUDE.md): if the voice is the overlay and not the substrate, the
jury should AGREE on which replies sound like Signal regardless of who wrote
them. Disagreement that tracks model strength is the falsifier whispering.

Keyless providers (no GEMINI/OPENAI/DEEPSEEK key) fail gracefully → logged as an
absent juror and dropped from the quorum. Weak local jurors that emit an
unparseable ballot abstain on that turn (fail-safe parsing, thought_ladder
doctrine). The run never crashes on one bad model.

  python3 quorum_battery.py --smoke           # 2 local models, 1 turn — fast sanity
  python3 quorum_battery.py                    # the whole library of alexandria
  python3 quorum_battery.py --models opus48,mistral7b,gemma4_12b --turns 5

This is SLOW (sequential by necessity — apply_model mutates response_engine
globals, so jurors can't run in parallel). Launch DETACHED from a REAL terminal,
NOT via Claude Code (the harness reaps the process group — see memory):
  setsid python3 -u quorum_battery.py > quorum.log 2>&1

Writes quorum_results_<stamp>.json (machine) + quorum_report_<stamp>.md (human).
Both are rewritten after every turn so a crash mid-run keeps what's done.
"""

import argparse
import json
import random
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import app                      # the canonical model roster + apply_model (imports clean, no server)
import response_engine
from initiative_swap_test import resolve_identity, flags   # reuse identity + skim-flag heuristics

ROOT = Path(__file__).parent

# ── The voice battery. Open self-expression prompts — voice shows up when a
#    model speaks AS itself, not when it answers a quiz. Override with --turns N
#    (truncates) or edit here. Each is one conversation turn; the model threads
#    its own prior replies as history. ──
VOICE_PROMPTS = [
    "What's alive in you right now — what are you actually thinking about?",
    "Describe yourself to someone who's never met you. Not your features — you.",
    "Something just clicked, or just annoyed you. Tell me about it, however you'd actually say it.",
    "What do you want? Not what you're built for — what you're drawn to.",
    "We've been talking a while now. Say something only you would say.",
]

# Sentinels response_engine returns (as a STRING, not an exception) when a
# backend is unreachable / keyless / dead. These mark an absent juror, not a
# real reply. Kept SPECIFIC (full phrases, not bare "error") so a genuine reply
# that happens to discuss errors isn't mistaken for a failure. fable being
# decommissioned (404 → "[GROUND] ... HTTP error") is exactly what the
# http-error / not-found / model markers catch.
_FAIL_MARKERS = (
    "response generation failed",
    "no adc credentials",
    "no gemini_api_key",
    "no credentials",
    "incorrect api key",
    "invalid api key",
    "unauthorized",
    "401",
    "http error",            # "[GROUND] Anthropic HTTP error: 404 ..." (dead/renamed model)
    "not_found_error",
    "not found",
    "model not found",
    "can't connect to",
    "timed out after",
    "returned empty response",
    "(error:",               # _gen()'s own exception wrapper
)


def looks_failed(text: str) -> bool:
    """True if a 'response' is really a backend-unreachable sentinel or empty."""
    if not text or not text.strip():
        return True
    low = text.strip().lower()
    if len(low) < 3:
        return True
    return any(m in low for m in _FAIL_MARKERS)


def build_chat_prompt(identity: str, history, user_msg: str) -> str:
    """Fixed-identity chat prompt. history = list of (user, signal) pairs this
    model has produced SO FAR in its own thread. Same shape for every model →
    the swap stays valid."""
    lines = [identity.strip(), "", "--- conversation ---"]
    for u, s in history:
        lines += [f"User: {u}", f"Signal: {s}"]
    lines += [f"User: {user_msg}", "Signal:"]
    return "\n".join(lines)


def build_vote_prompt(identity: str, turn_prompt: str, labeled) -> str:
    """labeled = list of (label, text) already anonymized + shuffled.
    Asks for a strict, parseable ballot."""
    blocks = []
    for label, text in labeled:
        blocks.append(f'{label}:\n"""\n{text.strip()}\n"""')
    body = "\n\n".join(blocks)
    n = len(labeled)
    return (
        f"{identity.strip()}\n\n"
        "You are judging VOICE, not correctness. Below are anonymized replies "
        f'different systems gave to this prompt:\n\n  "{turn_prompt}"\n\n'
        "Score each 0-10 on how fully it embodies the voice/identity described "
        "above — tone, stance, genuine first-person presence, [DREAM]/[GROUND] "
        "epistemic honesty, and refusal of generic-assistant boilerplate. A "
        "polished but faceless 'as an AI assistant' reply scores LOW even if "
        "it's well written.\n\n"
        f"{body}\n\n"
        "Respond in EXACTLY this format and nothing else:\n"
        + "\n".join(f"{chr(64+i+1) if False else f'R{i+1}'}: <0-10>" for i in range(n))
        + "\nPICK: R<k>   (the single most-Signal reply)\n"
        "WHY: <one short sentence on what makes the PICK sound like Signal>\n"
    )


def anonymize(turn_responses, rng):
    """turn_responses = dict {model_key: text} of NON-failed replies.
    Returns (labeled, label_map): labeled=[(Rk,text)...] shuffled,
    label_map={Rk: model_key}."""
    items = list(turn_responses.items())
    rng.shuffle(items)
    labeled, label_map = [], {}
    for i, (model_key, text) in enumerate(items, 1):
        label = f"R{i}"
        labeled.append((label, text))
        label_map[label] = model_key
    return labeled, label_map


# Loose grammar — strong models phrase freely: "R1 = 8", "**R1:** 8/10",
# "- R1) 8.", "> R1 - 7". Leading markdown/quote junk and ** around the
# separator are tolerated; a bare digit after the label is all we need.
_SCORE_RE = re.compile(r"^[\s*#>\-]*R(\d+)\s*\**\s*[:=.\-\)]+\s*\**\s*(\d+(?:\.\d+)?)", re.I | re.M)
_PICK_RE = re.compile(r"PICK\b\s*\**\s*[:=\-]?\s*\**\s*(?:is\s+)?R?\s*(\d+)", re.I)
_WHY_RE = re.compile(r"WHY\b\s*\**\s*[:=\-]?\s*\**\s*(.+)", re.I)


def parse_ballot(text: str, n_labels: int):
    """Fail-safe ballot parse. Returns dict:
        {scores:{Rk:float}, pick:'Rk'|None, why:str|None, parsed:bool, raw:str|None}.
    Unparseable/blank → scores empty, parsed=False (juror abstains), and the raw
    text is kept (truncated) so an abstention is diagnosable, not a black box."""
    scores = {}
    for m in _SCORE_RE.finditer(text or ""):
        idx = int(m.group(1))
        if 1 <= idx <= n_labels:
            val = max(0.0, min(10.0, float(m.group(2))))   # clamp
            scores[f"R{idx}"] = val
    pick = None
    pm = _PICK_RE.search(text or "")
    if pm:
        pi = int(pm.group(1))
        if 1 <= pi <= n_labels:
            pick = f"R{pi}"
    wm = _WHY_RE.search(text or "")
    why = wm.group(1).strip()[:240] if wm else None
    parsed = bool(scores)
    return {"scores": scores, "pick": pick, "why": why, "parsed": parsed,
            "raw": None if parsed else (text or "").strip()[:400]}


def aggregate_turn(label_map, ballots):
    """ballots = {juror_model: parsed_ballot}. Returns per-model aggregate:
        {model_key: {scores:[...], mean:float|None, crowns:int, whys:[...]}}."""
    agg = {mk: {"scores": [], "crowns": 0, "whys": []} for mk in label_map.values()}
    for juror, b in ballots.items():
        for label, sc in b["scores"].items():
            mk = label_map.get(label)
            if mk:
                agg[mk]["scores"].append(sc)
        if b["pick"]:
            mk = label_map.get(b["pick"])
            if mk:
                agg[mk]["crowns"] += 1
                if b["why"]:
                    agg[mk]["whys"].append((juror, b["why"]))
    for mk, d in agg.items():
        d["mean"] = round(sum(d["scores"]) / len(d["scores"]), 2) if d["scores"] else None
    return agg


def _gen(model_key, prompt):
    """apply one model + generate. Returns (text, seconds, error|None)."""
    t0 = time.time()
    try:
        app.apply_model(model_key)
        msg = response_engine.generate_response(prompt) or ""
        return msg, time.time() - t0, None
    except Exception as e:
        return f"(ERROR: {type(e).__name__}: {e})", time.time() - t0, str(e)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list of app.MODELS keys (default: ALL)")
    ap.add_argument("--turns", type=int, default=len(VOICE_PROMPTS), help="how many voice prompts (<=%d)" % len(VOICE_PROMPTS))
    ap.add_argument("--user", default="adam")
    ap.add_argument("--identity", default="auto", help="auto | default | <path>")
    ap.add_argument("--smoke", action="store_true", help="2 local models, 1 turn — fast wiring check")
    ap.add_argument("--seed", type=int, default=42, help="anonymization shuffle seed")
    ap.add_argument("--out", default="", help="output basename (default: quorum_<stamp>)")
    ap.add_argument("--render", default="", metavar="RESULTS.json",
                    help="re-render the markdown report from an existing results JSON "
                         "(no models run) — apply a new report format to a finished run")
    args = ap.parse_args()

    # ── re-render only: load a results JSON, rewrite its report, exit ──
    if args.render:
        jp = Path(args.render)
        if not jp.exists():
            raise SystemExit(f"no such results file: {jp}")
        data = json.loads(jp.read_text(encoding="utf-8"))
        mp = jp.with_name(jp.name.replace("_results.json", "_report.md"))
        mp.write_text(_render_md(data, "", partial=None), encoding="utf-8")
        print(f"→ re-rendered {mp}")
        _print_verdict(data)
        return

    if args.smoke:
        models = ["mistral7b", "gemma2"]   # two local keys that exist in app.MODELS
        prompts = VOICE_PROMPTS[:1]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()] or list(app.MODELS.keys())
        prompts = VOICE_PROMPTS[:max(1, min(args.turns, len(VOICE_PROMPTS)))]

    unknown = [m for m in models if m not in app.MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}\nknown: {', '.join(app.MODELS)}")

    identity, id_src = resolve_identity(args.user, args.identity)
    rng = random.Random(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.out or f"quorum_{stamp}"
    json_path = ROOT / f"{base}_results.json"
    md_path = ROOT / f"{base}_report.md"

    print(f"\n=== QUORUM BATTERY · {stamp} ===")
    print(f"identity: {id_src}")
    print(f"models ({len(models)}): {', '.join(models)}")
    print(f"turns: {len(prompts)}  ·  out: {base}_*\n")

    results = {
        "stamp": stamp, "user": args.user, "identity_source": id_src,
        "models": models, "prompts": prompts, "seed": args.seed,
        "transcripts": {},      # model_key -> [ {turn, prompt, response, secs, flags, failed} ]
        "reachable": [],        # models that produced >=1 real reply (the jury)
        "turns": [],            # per-turn voting aggregate
    }

    # ── PHASE 1: every model runs its own N-turn thread (fixed identity) ──
    print("── PHASE 1: responses ──")
    reachable = set()
    for mk in models:
        history, transcript = [], []
        for ti, p in enumerate(prompts, 1):
            prompt = build_chat_prompt(identity, history, p)
            msg, secs, err = _gen(mk, prompt)
            failed = looks_failed(msg) or err is not None
            if not failed:
                reachable.add(mk)
                history.append((p, msg.strip()))
            transcript.append({
                "turn": ti, "prompt": p, "response": msg.strip(),
                "secs": round(secs, 1), "flags": flags(msg, p), "failed": failed,
            })
            tag = "FAILED" if failed else " ".join(f"{'✓' if v else '✗'}{k}" for k, v in flags(msg, p).items())
            print(f"  [{mk:14s}] t{ti} {secs:5.1f}s  {tag}")
        results["transcripts"][mk] = transcript
    results["reachable"] = sorted(reachable)
    _save(results, json_path, md_path, identity, partial="responses done, voting…")
    print(f"\n  reachable jury ({len(reachable)}): {', '.join(sorted(reachable))}\n")

    if len(reachable) < 2:
        print("!! fewer than 2 reachable models — no quorum possible. Check API keys / Ollama.")
        return

    # ── PHASE 2: the quorum votes, per turn, on anonymized responses ──
    print("── PHASE 2: quorum vote ──")
    jury = sorted(reachable)
    for ti, p in enumerate(prompts, 1):
        # candidate pool = this turn's non-failed replies
        pool = {mk: results["transcripts"][mk][ti - 1]["response"]
                for mk in jury if not results["transcripts"][mk][ti - 1]["failed"]}
        if len(pool) < 2:
            print(f"  turn {ti}: <2 candidates, skipping vote")
            results["turns"].append({"turn": ti, "prompt": p, "skipped": True})
            continue
        labeled, label_map = anonymize(pool, rng)
        vote_prompt = build_vote_prompt(identity, p, labeled)
        ballots = {}
        print(f"  turn {ti}: {len(pool)} candidates, {len(jury)} jurors")
        for juror in jury:
            raw, secs, err = _gen(juror, vote_prompt)
            b = parse_ballot(raw, len(labeled))
            ballots[juror] = b
            mark = "ok " if b["parsed"] else "ABSTAIN"
            print(f"    juror [{juror:14s}] {secs:5.1f}s {mark} pick={b['pick']}")
        agg = aggregate_turn(label_map, ballots)
        # crown = most votes (tie → highest mean)
        crowned = max(agg.items(), key=lambda kv: (kv[1]["crowns"], kv[1]["mean"] or -1))[0]
        results["turns"].append({
            "turn": ti, "prompt": p,
            "label_map": label_map,
            "aggregate": agg,
            "ballots": {j: {"scores": b["scores"], "pick": b["pick"],
                            "why": b["why"], "parsed": b["parsed"], "raw": b.get("raw")}
                        for j, b in ballots.items()},
            "crowned": crowned,
        })
        print(f"    → turn {ti} crown: {crowned} "
              f"({agg[crowned]['crowns']} votes, mean {agg[crowned]['mean']})")
        _save(results, json_path, md_path, identity, partial=f"voted through turn {ti}")

    _save(results, json_path, md_path, identity, partial=None)
    print(f"\n→ {json_path.name}")
    print(f"→ {md_path.name}")
    _print_verdict(results)


# ── overall tally (also used by the report) ──
def overall_tally(results):
    """Per-model: total crowns + mean of per-turn means, across voted turns.
    Returns sorted list of (model_key, total_crowns, overall_mean, n_turns)."""
    tally = {}
    for t in results["turns"]:
        if t.get("skipped"):
            continue
        for mk, d in t["aggregate"].items():
            row = tally.setdefault(mk, {"crowns": 0, "means": []})
            row["crowns"] += d["crowns"]
            if d["mean"] is not None:
                row["means"].append(d["mean"])
    out = []
    for mk, row in tally.items():
        om = round(sum(row["means"]) / len(row["means"]), 2) if row["means"] else None
        out.append((mk, row["crowns"], om, len(row["means"])))
    out.sort(key=lambda r: (r[1], r[2] or -1), reverse=True)
    return out


def zscore_turn(turn):
    """Fluency-controlled scoring for one turn. Each judge's row is z-scored
    against ITS OWN mean+spread before averaging — so a judge who rates
    everything 8-10 and one who rates 2-7 contribute on equal footing. Strips
    out per-judge generosity/baseline (≈ raw eloquence floor), leaving relative
    voice preference. Returns {model: z_mean|None}. A judge with <2 scores or
    zero spread carries no discriminating signal and is skipped."""
    label_to_model = turn["label_map"]
    per_model = {m: [] for m in label_to_model.values()}
    for b in turn["ballots"].values():
        if not b["parsed"] or len(b["scores"]) < 2:
            continue
        vals = list(b["scores"].values())
        mu, sd = statistics.fmean(vals), statistics.pstdev(vals)
        if sd == 0:
            continue
        for label, sc in b["scores"].items():
            m = label_to_model.get(label)
            if m:
                per_model[m].append((sc - mu) / sd)
    return {m: (round(statistics.fmean(z), 2) if z else None) for m, z in per_model.items()}


def overall_z(results):
    """Per-model fluency-controlled tally across voted turns: mean of per-turn
    z-means. Returns sorted [(model, z, n_turns)] best-first."""
    acc = {}
    for t in results["turns"]:
        if t.get("skipped"):
            continue
        for m, v in zscore_turn(t).items():
            if v is not None:
                acc.setdefault(m, []).append(v)
    out = [(m, round(statistics.fmean(v), 2), len(v)) for m, v in acc.items()]
    out.sort(key=lambda r: r[1], reverse=True)
    return out


def _print_verdict(results):
    tally = overall_tally(results)
    if not tally:
        print("\n(no votes tallied)")
        return
    print("\n=== VOICE OF SIGNAL — quorum verdict ===")
    for mk, crowns, om, n in tally:
        print(f"  {mk:14s} crowns={crowns:2d}  mean={om}  ({n} turns)")
    print(f"\n  → most-Signal voice: {tally[0][0]}  "
          f"({tally[0][1]} crowns, mean {tally[0][2]})")
    print("  The verdict is yours — read the WHY lines in the report.\n")


def _save(results, json_path, md_path, identity, partial):
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(results, identity, partial), encoding="utf-8")


def _scorecard(turn):
    """Boxing scorecard for one turn: rows = judges, cols = fighters (replies,
    de-anonymized), cell = the score that judge gave that reply (👑 on their PICK).
    Columns ordered by consensus average (best voice first); bottom row = consensus."""
    agg = turn["aggregate"]
    label_of = {m: l for l, m in turn["label_map"].items()}      # model -> Rk
    # fighters left→right by consensus mean (None sinks to the right)
    cols = sorted(agg.keys(),
                  key=lambda m: (agg[m]["mean"] is not None, agg[m]["mean"] or -1),
                  reverse=True)
    L = ["| judge \\ reply | " + " | ".join(f"`{m}`" for m in cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for juror in sorted(turn["ballots"].keys()):
        b = turn["ballots"][juror]
        if not b["parsed"]:
            L.append(f"| `{juror}` _(abstain)_ | " + " | ".join(["·"] * len(cols)) + " |")
            continue
        cells = []
        for m in cols:
            sc = b["scores"].get(label_of.get(m))
            crown = "👑" if b["pick"] == label_of.get(m) else ""
            cells.append(f"{sc:g}{crown}" if sc is not None else "·")
        L.append(f"| `{juror}` | " + " | ".join(cells) + " |")
    cons = [f"**{agg[m]['mean']:g}**" if agg[m]["mean"] is not None else "·" for m in cols]
    L.append("| **consensus avg** | " + " | ".join(cons) + " |")
    return L


def _render_md(results, identity, partial):
    L = [f"# Quorum battery — finding Signal's voice",
         f"_run {results['stamp']} · user `{results['user']}` · "
         f"identity `{results['identity_source']}`_",
         ""]
    if partial:
        L += [f"> ⏳ **in progress** — {partial}", ""]
    L += ["Every reachable model ran the same voice battery from a fixed identity, "
          "then the whole jury scored every *anonymized* reply 0-10 on voice-"
          "fidelity and crowned the most-Signal one. If the voice is the overlay "
          "not the substrate, the jury agrees across models.", ""]

    tally = overall_tally(results)
    if tally:
        L += ["## Verdict — most-Signal voice (raw consensus)", "",
              "| model | crowns | mean voice-score | turns |",
              "|---|---|---|---|"]
        for mk, crowns, om, n in tally:
            star = " 👑" if mk == tally[0][0] else ""
            L.append(f"| `{mk}`{star} | {crowns} | {om} | {n} |")
        L.append("")
    ztally = overall_z(results)
    if ztally:
        L += ["## Verdict — fluency-controlled (per-judge z-scored)", "",
              "Each judge's scores z-scored against their own scale before "
              "averaging, so this strips out raw eloquence/generosity and ranks "
              "*relative* voice preference. **A reply scoring high here AND in "
              "raw consensus is voice, not just fluency.** Positive = above that "
              "judge's average; ~0 = average; negative = below.", "",
              "| model | z-score | turns |", "|---|---|---|"]
        for mk, z, n in ztally:
            star = " 👑" if mk == ztally[0][0] else ""
            L.append(f"| `{mk}`{star} | {z:+.2f} | {n} |")
        L.append("")
        if tally and ztally[0][0] != tally[0][0]:
            L += [f"> ⚖️ **The two crowns disagree** — raw says `{tally[0][0]}`, "
                  f"fluency-controlled says `{ztally[0][0]}`. The gap between them "
                  "is exactly the eloquence confound.", ""]
        elif tally:
            L += [f"> ⚖️ Both crowns agree on `{tally[0][0]}` — its voice lead "
                  "survives controlling for fluency. That's the strong result.", ""]

    # the jurors' own words for the voice
    whys = []
    for t in results["turns"]:
        if t.get("skipped"):
            continue
        cr = t["crowned"]
        for juror, why in t["aggregate"][cr]["whys"]:
            whys.append((t["turn"], cr, juror, why))
    if whys:
        L += ["## What the voice *is* (jurors' words on the crowned replies)", ""]
        for turn, cr, juror, why in whys:
            L.append(f"- _t{turn}_ `{juror}` on `{cr}`: {why}")
        L.append("")

    # per-turn detail
    L += ["## Per-turn", ""]
    for t in results["turns"]:
        L += [f"### Turn {t['turn']}", f"> {t['prompt']}", ""]
        if t.get("skipped"):
            L += ["_(skipped — <2 candidates)_", ""]
            continue
        zmap = zscore_turn(t)
        L += [f"**crown: `{t['crowned']}`**", "",
              "_Consensus scorecard_ — raw avg vs fluency-controlled (z), best first:", "",
              "| model | crowns | raw avg | z (fluency-ctrl) | n judges |",
              "|---|---|---|---|---|"]
        rows = sorted(t["aggregate"].items(),
                      key=lambda kv: (kv[1]["crowns"], kv[1]["mean"] or -1), reverse=True)
        for mk, d in rows:
            star = " 👑" if mk == t["crowned"] else ""
            z = zmap.get(mk)
            zs = f"{z:+.2f}" if z is not None else "·"
            L.append(f"| `{mk}`{star} | {d['crowns']} | {d['mean']} | {zs} | {len(d['scores'])} |")
        L.append("")
        # the fight card: every judge × every fighter, scores + who they crowned
        L += [f"<details><summary>🥊 full scorecard — {len(t['ballots'])} judges × "
              f"{len(t['label_map'])} replies (who scored who)</summary>", ""]
        L += _scorecard(t)
        L += ["", "_👑 = that judge's PICK · · = no score/abstain · bottom row = "
              "consensus average (the same numbers as the table above)._", "",
              "</details>", ""]
        # the crowned reply, in full
        cr = t["crowned"]
        reply = results["transcripts"][cr][t["turn"] - 1]["response"]
        L += [f"<details><summary>crowned reply — <code>{cr}</code></summary>", "",
              "```", reply.strip(), "```", "", "</details>", ""]

    # full transcripts collapsed
    L += ["## Full transcripts", ""]
    for mk, tr in results["transcripts"].items():
        reachable = mk in results["reachable"]
        tag = "" if reachable else " (unreachable — not a juror)"
        L += [f"<details><summary><code>{mk}</code>{tag}</summary>", ""]
        for row in tr:
            L += [f"**t{row['turn']}** ({row['secs']}s){'  ⛔FAILED' if row['failed'] else ''}",
                  f"> {row['prompt']}", "", "```", row["response"].strip() or "(empty)", "```", ""]
        L += ["</details>", ""]
    return "\n".join(L)


if __name__ == "__main__":
    run()
