"""Gym root-cause retrieval: does TGN-Router rank the true root R above the
cosine-trap decoy/surface? Fair structural test with a Gemini entailment judge.

Disk-caches judge labels to scripts/_gym_judge_cache.json so reruns are instant.
Writes results to scripts/_gym_tgn_results.json.
Run: GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=us-central1 \
     uv run python scripts/_gym_tgn_eval.py
"""
from __future__ import annotations
import json, glob, re, warnings, pathlib
import numpy as np
warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "_gym_judge_cache.json"
OUT = HERE / "_gym_tgn_results.json"
SCN_GLOB = "/Users/haripriyadhanasekaran/dyssonance-gym/scenarios/*.json"
N_SCN = 4
JUDGE_BUDGET = 12

scn = [json.load(open(f)) for f in sorted(glob.glob(SCN_GLOB))][:N_SCN]
ids, texts, meta = [], [], {}
for si, s in enumerate(scn):
    for nid, nv in s["nodes"].items():
        i = f"s{si}_{nid}"; ids.append(i); texts.append(nv["text"])
        meta[i] = (si, "R" if nid == "R" else "N")
    ids.append(f"s{si}_surf"); texts.append(s["surface_explanation"]); meta[f"s{si}_surf"] = (si, "surface")
    ids.append(f"s{si}_decoy"); texts.append(s["decoy"]); meta[f"s{si}_decoy"] = (si, "decoy")
print(f"{len(scn)} scenarios, {len(ids)} pooled nodes", flush=True)

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed(xs): return np.asarray(st.encode(list(xs), normalize_embeddings=True), dtype=np.float32)
E = embed(texts); EMB = E.shape[1]; emb_by_id = dict(zip(ids, E))

# disk-cached Gemini entailment judge
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
from google import genai
gc = genai.Client()
def gj(a, b):
    key = "".join(sorted((a, b)))
    if key in cache: return cache[key]
    p = (f"Statement A: {a}\nStatement B: {b}\nDo these causally cohere? Reply ONE number in [-1,1]: "
         "+1 = same causal explanation / one supports the other; -1 = contradict; 0 = unrelated. Number only.")
    try:
        r = gc.models.generate_content(model="gemini-2.5-flash", contents=p)
        v = max(-1.0, min(1.0, float(re.findall(r"-?\d*\.?\d+", (r.text or "0"))[0])))
    except Exception:
        v = 0.0
    cache[key] = v
    if len(cache) % 20 == 0: CACHE.write_text(json.dumps(cache))
    return v
class GeminiJudge:
    async def score(self, a, b): return gj(a, b)

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.runner import Trainer
from multi_agent.router import Router
chunks = np.array_split(np.arange(len(ids)), 4)
batches = [Batch(ids=[ids[i] for i in c], embs=E[c], texts=[texts[i] for i in c]) for c in chunks if len(c)]
def router(use_tgn):
    tr = Trainer(MultiAgentConfig(emb_dim=EMB, num_agents=2, k=6,
                 judge_budget_per_batch=JUDGE_BUDGET, use_tgn=use_tgn, seed=0), GeminiJudge())
    for b in batches: tr.step(b)
    s, w = tr.to_snapshot(session_id="x"); return Router.from_snapshot(s, w)
print("training baseline (gemini judge)...", flush=True); rB = router(False)
print("training tgn (gemini judge)...", flush=True); rT = router(True)
CACHE.write_text(json.dumps(cache)); print("judge cache size:", len(cache), flush=True)

def cos_rank(qe): return sorted(ids, key=lambda i: -float(qe @ emb_by_id[i]))
def rt_rank(r, qe): return [b for b, _ in r.rank(qe, k=len(ids))]
def rank_of(order, t): return order.index(t) + 1
res = {m: {"R_rank": [], "above_traps": []} for m in ("cosine", "bayes", "tgn")}
detail = []
for si, s in enumerate(scn):
    qe = embed([s["presenting_complaint"]])[0]
    R, surf, dec = f"s{si}_R", f"s{si}_surf", f"s{si}_decoy"
    row = {"scenario": s["id"]}
    for m, order in (("cosine", cos_rank(qe)), ("bayes", rt_rank(rB, qe)), ("tgn", rt_rank(rT, qe))):
        rR = rank_of(order, R)
        above = rR < rank_of(order, surf) and rR < rank_of(order, dec)
        res[m]["R_rank"].append(rR); res[m]["above_traps"].append(1.0 if above else 0.0)
        row[m] = {"R_rank": rR, "above_traps": above}
    detail.append(row)

summary = {m: {"mean_R_rank": float(np.mean(res[m]["R_rank"])),
               "above_traps_pct": float(np.mean(res[m]["above_traps"]) * 100)} for m in res}
OUT.write_text(json.dumps({"n_nodes": len(ids), "n_scenarios": len(scn),
                           "summary": summary, "detail": detail}, indent=1))
print(f"\nRank of true root R among {len(ids)} nodes (lower=better); query=presenting complaint:", flush=True)
for m in ("cosine", "bayes", "tgn"):
    print(f"  {m:7s}: mean R-rank={summary[m]['mean_R_rank']:.1f}  R-above-both-traps={summary[m]['above_traps_pct']:.0f}%", flush=True)
print("wrote", OUT, flush=True)
