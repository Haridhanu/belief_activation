"""Pooled, multi-episode AMABench substrate A/B with an OpenAI entailment judge.

Held-out signed link-prediction AUC: cosine vs Bayesian 2-hop vs SignedGNN, on
real AMABench belief text with LLM-judged coherence/contradiction labels.
Concurrent judging + disk cache; multi-seed splits for mean±std.
"""
from __future__ import annotations
import json, re, pathlib, itertools, warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

DS = "/Users/haripriyadhanasekaran/.cache/huggingface/hub/datasets--AMA-bench--AMA-bench/snapshots/a5777378066f53229a94557a7b192435cd027909/test/open_end_qa_set.jsonl"
N_EP = 5            # episodes to pool
TURNS = 36         # belief nodes per episode
PAIRS_PER_EP = 450 # pairs to judge per episode (temporal-distance-biased)
CACHE = pathlib.Path(__file__).resolve().parent / "_ama_judge_cache.json"

rows = [json.loads(l) for l in open(DS)]
eps = [r for r in rows if r["domain"] != "Game" and len(r["trajectory"]) >= TURNS][:N_EP]
print("episodes:", [(e["episode_id"], e["task_type"], len(e["trajectory"])) for e in eps], flush=True)

def turn_text(t):
    obs = " ".join(str(t.get("observation", "")).split())[:220]
    return f"Step {t['turn_idx']}: action={t.get('action','')} | {obs}"

# build pooled node list (per-episode offset) + texts
node_texts, node_ep, ep_nodes = [], [], {}
for ei, e in enumerate(eps):
    tt = [turn_text(t) for t in e["trajectory"][:TURNS]]
    base = len(node_texts)
    ep_nodes[ei] = list(range(base, base + len(tt)))
    node_texts += tt
    node_ep += [ei] * len(tt)
N = len(node_texts)

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = np.asarray(st.encode(node_texts, normalize_embeddings=True), dtype=np.float32)
EMB = X.shape[1]
print(f"pooled nodes: {N}, emb_dim {EMB}", flush=True)

from openai import OpenAI
oc = OpenAI()
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

def judge(i, j):
    k = f"p_{min(i,j)}_{max(i,j)}"
    if k in cache:
        return cache[k]
    a, b = node_texts[i], node_texts[j]
    p = (f"Two events from one agent trajectory.\nA: {a}\nB: {b}\n"
         "Reply ONE number in [-1,1]: +1 if consistent / one supports or follows the other; "
         "-1 if they contradict (a state in one is negated or changed in the other); 0 if unrelated. "
         "Number only.")
    try:
        r = oc.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": p}], max_tokens=5, temperature=0)
        v = max(-1.0, min(1.0, float(re.findall(r"-?\d*\.?\d+", r.choices[0].message.content)[0])))
    except Exception:
        v = 0.0
    cache[k] = v
    return v

# temporal-distance-biased within-episode pair sampling
rng = np.random.default_rng(0)
to_judge = []
for ei, nodes in ep_nodes.items():
    pairs = list(itertools.combinations(nodes, 2))
    w = np.array([abs(i - j) for i, j in pairs], dtype=float); w /= w.sum()
    idx = rng.choice(len(pairs), size=min(PAIRS_PER_EP, len(pairs)), replace=False, p=w)
    to_judge += [pairs[k] for k in idx]
print(f"judging {len(to_judge)} pairs (concurrent)...", flush=True)
with ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(lambda ij: judge(*ij), to_judge))
CACHE.write_text(json.dumps(cache))

labeled = [(i, j, 1 if judge(i, j) > 0 else -1) for i, j in to_judge if abs(judge(i, j)) >= 0.5]
npos = sum(1 for *_, l in labeled if l > 0); nneg = len(labeled) - npos
print(f"signed pairs: {len(labeled)}  (+{npos} / -{nneg})", flush=True)

from multi_agent.graph import Graph
from multi_agent.signed_gnn import SignedGNN
import torch

def auc(s, y):
    s = np.asarray(s); y = np.asarray(y); p = s[y == 1]; n = s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

ids = [f"n{i}" for i in range(N)]
results = {"cosine": [], "bayesian": [], "signed": []}
for seed in range(5):
    r2 = np.random.default_rng(seed)
    lab = labeled[:]; r2.shuffle(lab)
    cut = int(0.7 * len(lab)); obs, hold = lab[:cut], lab[cut:]
    y = [1 if l > 0 else 0 for *_, l in hold]
    g = Graph(emb_dim=EMB)
    g.extend(ids, X, [(ids[i], ids[j], float(l)) for i, j, l in obs])
    results["cosine"].append(auc([float(X[i] @ X[j]) for i, j, _ in hold], y))
    results["bayesian"].append(auc([g.field(ids[i], ids[j]) for i, j, _ in hold], y))
    m = SignedGNN(EMB, 32, 3)
    m.fit(X, [(i, j, float(l)) for i, j, l in obs], epochs=250, lr=0.01)
    sp = m.predict(X, [(i, j) for i, j, _ in hold])
    results["signed"].append(auc([sp[(i, j)] for i, j, _ in hold], y))

print(f"\nHeld-out link-pred AUC over 5 seeds (pooled {N_EP} alfworld episodes, {len(labeled)} signed pairs):")
for k, v in results.items():
    v = [x for x in v if x == x]
    print(f"  {k:9s}: {np.mean(v):.3f} +/- {np.std(v):.3f}")
print("(tgn substrate ~0.51 from the faithful Trainer run — needs the PSRO loop, omitted here)")
