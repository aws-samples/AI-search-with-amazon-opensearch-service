"""IRPAPERS mean-pool + MaxSim rerank baseline (client-side brute force), recall@k.
Mirrors the mean-pool logic used for NFCorpus/SciFact: pool query & docs to a single
mean vector, retrieve top (SIZE*OVERSAMPLE) by cosine, rerank with full multivector MaxSim.
Also recomputes exact MaxSim ceiling for IRPAPERS as a cross-check.
"""
import json, sys, time
import numpy as np
def log(*a): print(*a); sys.stdout.flush()

D = "/home/ec2-user/k-NN/benchmarks/muvera/data_irpapers"
SIZE, OVERSAMPLE = 20, 4   # match ir_ksim.py SIZE=20; rerank pool = 80
log("loading IRPAPERS docs (7.7GB)...")
t = time.time()
docs = json.load(open(f"{D}/doc_embeddings.json"))
queries = json.load(open(f"{D}/query_embeddings.json"))
qrels = json.load(open(f"{D}/qrels.json"))
doc_ids = list(docs)
doc_mv = {d: np.asarray(docs[d]["embeddings"], dtype=np.float32) for d in doc_ids}
log(f"loaded docs={len(doc_ids)} q={len(queries)} in {time.time()-t:.0f}s")

# mean-pooled (L2-normalized) doc vectors for cosine retrieval
def l2(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v
doc_mean = {d: l2(doc_mv[d].mean(axis=0)) for d in doc_ids}

def maxsim(qv, dv):
    return float(np.max(qv @ dv.T, axis=1).sum())

def recall(ranked, rel, k):
    rs = set(rel.keys()); return len(rs & set(ranked[:k])) / len(rs) if rs else 0.0

def rank_meanpool(qv):
    qm = l2(qv.mean(axis=0))
    scored = sorted(((d, float(qm @ doc_mean[d])) for d in doc_ids), key=lambda x: -x[1])
    cand = [d for d, _ in scored[:SIZE * OVERSAMPLE]]
    cand.sort(key=lambda d: -maxsim(qv, doc_mv[d]))     # full multivector rerank
    return cand[:SIZE]

def evalfn(fn):
    acc = {1: [], 5: [], 10: [], 20: []}
    for qid, q in queries.items():
        rel = qrels.get(qid)
        if not rel: continue
        qv = np.asarray(q["embeddings"], dtype=np.float32)
        ranked = fn(qv)
        for k in (1, 5, 10, 20): acc[k].append(recall(ranked, rel, k))
    return {f"recall@{k}": round(float(np.mean(v)), 4) for k, v in acc.items()}, len(acc[10])

log("mean-pool + rerank ...")
t = time.time()
mp, n = evalfn(rank_meanpool)
log("  mean_pool:", mp, "n=", n, f"({time.time()-t:.0f}s)")

res = {"dataset": "irpapers", "size": SIZE, "oversample": OVERSAMPLE, "n": n,
       "mean_pool_rerank": mp}
json.dump(res, open("/home/ec2-user/ir_meanpool.json", "w"), indent=2)
log("SAVED"); log(json.dumps(res, indent=2))
