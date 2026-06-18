#!/usr/bin/env python3
"""reverse_cross.py — v1 model on v2's held-out exam (leakage-guarded).
Completes the 2x2: tells us if v2's lower score is a convention mismatch
(v1 also tanks here) or genuine v2-data quality (v1 does fine here)."""
import json, random
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from canon_verbs import CANON

SEED, FRAC = 42, 0.20
def split(path):
    rows=[json.loads(l) for l in open(path) if l.strip()]
    rng=random.Random(SEED); sh=rows[:]; rng.shuffle(sh)
    ne=int(len(sh)*FRAC); return sh[ne:], sh[:ne]

v1_train,_ = split("omni_phase1_corpus.jsonl")
_, v2_eval = split("omni_phase1_corpus.v2.jsonl")
v1_train_eng={r["english"] for r in v1_train}
clean=[r for r in v2_eval if r["english"] not in v1_train_eng]
print(f"v2 held-out: {len(v2_eval)}  | in v1-train (leak): {len(v2_eval)-len(clean)}  | clean: {len(clean)}")

dev="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained("omniaddress_model")
m=T5ForConditionalGeneration.from_pretrained("omniaddress_model").to(dev).eval()
ok=cok=0; fields=[0]*5
for it in clean:
    inp=tok("omniaddress: "+it["english"],return_tensors="pt",max_length=96,truncation=True).to(dev)
    with torch.no_grad(): out=m.generate(**inp,max_new_tokens=24,num_beams=4)
    pred=tok.decode(out[0],skip_special_tokens=True).strip(); gold=it["omni"].strip()
    if pred==gold: ok+=1
    pf,gf=pred.split("."),gold.split(".")
    for i in range(min(5,len(gf))):
        if i<len(pf) and pf[i]==gf[i]: fields[i]+=1
    if len(pf)==5 and len(gf)==5:
        p,g=pf[:],gf[:]; p[1]=CANON.get(pf[1],pf[1]); g[1]=CANON.get(gf[1],gf[1])
        if p==g: cok+=1
n=len(clean); labels=["subject","verb","object","tense","negator"]
print(f"\nv1 model on v2 exam:  rich {ok}/{n}={100.0*ok/n:.1f}%   canon {cok}/{n}={100.0*cok/n:.1f}%")
print("  per-field: "+"  ".join(f"{l} {100.0*fc/n:.1f}" for l,fc in zip(labels,fields)))
