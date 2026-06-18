#!/usr/bin/env python3
"""
eval_omniaddress.py — measure exact-match hit rate of a trained OmniAddress model
on the SAME seeded held-out split train_omniaddress.py uses (SEED=42, 20%).

Self-contained (no `datasets` dependency) so it runs for pure inference.

Usage:
  python3 eval_omniaddress.py [model_path]
  default = omniaddress_model/checkpoints/checkpoint-120
"""
import sys
import os
import json
import random
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# These must match train_omniaddress.py exactly so the held-out 60 are identical
CORPUS_PATH    = Path(os.environ.get("OMNI_CORPUS", "omni_phase1_corpus.jsonl"))  # default v1; override via OMNI_CORPUS to match the model under eval.
EVAL_SPLIT     = 0.20
SEED           = 42
MAX_INPUT_LEN  = 96
MAX_TARGET_LEN = 24

model_path = sys.argv[1] if len(sys.argv) > 1 else "omniaddress_model"

def load_corpus(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs

def split_corpus(pairs, eval_frac, seed):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_frac)
    return shuffled[n_eval:], shuffled[:n_eval]   # train, eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[eval] model: {model_path}  device: {device}")

pairs = load_corpus(CORPUS_PATH)
_, eval_pairs = split_corpus(pairs, EVAL_SPLIT, SEED)
print(f"[eval] held-out examples: {len(eval_pairs)}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

correct = 0
field_correct = [0, 0, 0, 0, 0]   # per-position accuracy: subject.verb.object.tense.negator
for item in eval_pairs:
    inp = tokenizer("omniaddress: " + item["english"], return_tensors="pt",
                    max_length=MAX_INPUT_LEN, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_TARGET_LEN, num_beams=4)
    pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    gold = item["omni"].strip()
    if pred == gold:
        correct += 1
    # per-field diagnostics
    pf, gf = pred.split("."), gold.split(".")
    for i in range(min(5, len(gf))):
        if i < len(pf) and pf[i] == gf[i]:
            field_correct[i] += 1
    status = "✓" if pred == gold else "✗"
    print(f"  {status}  pred: {pred:<42} gold: {gold}")

n = len(eval_pairs)
pct = 100.0 * correct / n
labels = ["subject", "verb", "object", "tense", "negator"]
print(f"\n[eval] exact-match hit rate: {correct}/{n}  ({pct:.1f}%)")
print("[eval] per-field accuracy:")
for lab, fc in zip(labels, field_correct):
    print(f"        {lab:<8} {100.0*fc/n:5.1f}%")
