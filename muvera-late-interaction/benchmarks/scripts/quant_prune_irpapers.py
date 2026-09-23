"""
COMBINED int8 quantization x norm-pruning on IRPAPERS (dense docs, ~1000 vec/page).
Mirrors quant_prune_combined.py but uses IRPAPERS' metric: recall@10 against qrels.json
(|relevant ∩ top-k| / |relevant|), matching ir_ksim.py.

Same mechanics as the SciFact/NFCorpus combined runs so results are comparable:
  - client-side FDE (K_SIM=4, DIM_PROJ=16, R_REPS=40 -> FDE=10240, SEED=42) for phase-1 prefetch,
    full-token float32 FDE in every cell (candidate set identical across cells)
  - SIZE=10, OVERSAMPLE=4 -> 40 rerank candidates; lateInteractionScore reads colbert_vectors
  - int8: symmetric per-vector quantize->dequantize round-trip (stored dequantized in _source)
  - norm-based doc pruning: keep top-R% tokens per doc by L2 magnitude
  - true compact multivector MB = bytes_per(int8=1) * scalars_kept_after_prune

int8 at retention {100,75,50}%. Writes /home/ec2-user/quant_prune_irpapers.json

Run on EC2 in ~/k-NN/benchmarks/muvera with the 3.8 cluster (9210) up:
    python3 quant_prune_irpapers.py <max_docs|0=all>
"""
import json, math, sys, time
import numpy as np
from opensearchpy import OpenSearch, helpers

DATA = "data_irpapers"
MAXD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DIM, SEED = 128, 42
K_SIM, DIM_PROJ, R_REPS = 4, 16, 40
FDE = R_REPS * (1 << K_SIM) * DIM_PROJ
SIZE, OVERSAMPLE, SPACE = 10, 4, "innerproduct"
PORT = 9210
RETAIN = [25]
BYTES_PER_INT8 = 1
c = OpenSearch(hosts=[{"host": "127.0.0.1", "port": PORT}], use_ssl=False, timeout=1800)
def log(*a): print(*a); sys.stdout.flush()

class Fde:
    def __init__(s, dim, k, dp, r, seed):
        s.dim, s.k, s.dp, s.r = dim, k, dp, r; s.parts = 1 << k
        rng = np.random.RandomState(seed)
        s.planes = rng.randn(r, k, dim).astype(np.float32)
        s.proj = np.where(rng.rand(r, dp, dim) < 0.5, -1.0, 1.0).astype(np.float32)
        s.scale = 1.0 / math.sqrt(dp)
    def encode(s, vecs, is_doc):
        vecs = np.asarray(vecs, np.float32); n = vecs.shape[0]
        out = np.zeros(s.r * s.parts * s.dp, np.float32); off = 0
        for r in range(s.r):
            dots = vecs @ s.planes[r].T
            bits = (dots > 0).astype(np.int32)
            w = (1 << np.arange(s.k)).astype(np.int32); cids = bits @ w
            centers = np.zeros((s.parts, s.dim), np.float32); counts = np.zeros(s.parts, np.int32)
            for v in range(n):
                centers[cids[v]] += vecs[v]; counts[cids[v]] += 1
            if is_doc:
                nz = counts > 1; centers[nz] /= counts[nz][:, None]
                empties = np.where(counts == 0)[0]; nonempty = np.where(counts > 0)[0]
                if len(empties) and len(nonempty):
                    for cpart in empties:
                        d = np.array([bin(int(cpart) ^ int(o)).count("1") for o in nonempty])
                        nearest = nonempty[int(np.argmin(d))]
                        idx = int(np.where(cids == nearest)[0][0]); centers[cpart] = vecs[idx]
            projected = (centers @ s.proj[r].T) * s.scale
            out[off:off + projected.size] = projected.reshape(-1); off += projected.size
        return out

def q_int8(mv):
    out = []
    for v in mv:
        a = np.asarray(v, np.float32); amax = float(np.max(np.abs(a)))
        sc = amax / 127.0 if amax > 0 else 1.0
        qi = np.clip(np.round(a / sc), -127, 127).astype(np.int8)
        out.append((qi.astype(np.float32) * sc).tolist())
    return out

def recall(ranked, rel, k):
    rs = set(rel.keys())
    return len(rs & set(ranked[:k])) / len(rs) if rs else 0.0
def dcg(r): return sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(r))

D = f"/home/ec2-user/k-NN/benchmarks/muvera/{DATA}"
queries = json.load(open(f"{D}/query_embeddings.json"))
qrels = json.load(open(f"{D}/qrels.json"))
log(f"loading 7.7GB docs..."); t = time.time()
docs = json.load(open(f"{D}/doc_embeddings.json"))
log(f"loaded {len(docs)} docs in {time.time()-t:.0f}s")
if MAXD: docs = dict(list(docs.items())[:MAXD])

enc = Fde(DIM, K_SIM, DIM_PROJ, R_REPS, SEED)
doc_mv = {}; doc_fde = {}; doc_order = {}
t = time.time(); dt = []
for i, (did, d) in enumerate(docs.items()):
    mv = np.asarray(d["embeddings"], np.float32)
    doc_mv[did] = mv
    doc_fde[did] = enc.encode(mv, True).tolist()
    doc_order[did] = np.argsort(-np.linalg.norm(mv, axis=1))
    dt.append(len(mv))
    if (i + 1) % 500 == 0: log(f"  FDE {i+1}/{len(docs)} avg_tok={np.mean(dt):.0f} {time.time()-t:.0f}s")
del docs
q_fde = {qid: enc.encode(q["embeddings"], False).tolist() for qid, q in queries.items()}
log(f"client FDE precompute {time.time()-t:.0f}s; avg_doc_tokens={np.mean(dt):.1f}")

src = "lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)"
def body(qfde, qmv):
    return {"query": {"script_score": {"query": {"knn": {"muvera_fde": {"vector": qfde, "k": SIZE * OVERSAMPLE}}},
            "script": {"source": src, "params": {"query_vectors": qmv, "space_type": SPACE}}}},
            "size": SIZE, "_source": {"excludes": ["colbert_vectors", "muvera_fde"]}}

def q_float32(mv): return mv
def quant(mv, variant): return mv if variant == "float32" else q_int8(mv)
BYTES_PER = {"float32": 4, "int8": 1}
def mrr(ranked, rel, k):
    for i, d in enumerate(ranked[:k]):
        if d in rel: return 1.0 / (i + 1)
    return 0.0
def ndcg(ranked, rel, k):
    idcg = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(d, 0) for d in ranked[:k]]) / idcg if idcg > 0 else 0.0

# CELLS: float32 @100 (baseline) + int8 @25 (slide point). Both on identical FDE prefetch.
CELLS = [("float32", 100), ("int8", 25)]
results = {"data": DATA, "fde": FDE, "num_docs": len(doc_mv), "cells": {}}
for variant, R in CELLS:
    IDX = f"muvera-irqp-{variant}-{R}"
    if c.indices.exists(index=IDX): c.indices.delete(index=IDX)
    c.indices.create(index=IDX, body={"settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"colbert_vectors": {"type": "object", "enabled": False},
            "muvera_fde": {"type": "knn_vector", "dimension": FDE, "method": {"name": "hnsw", "engine": "lucene",
                "space_type": SPACE, "parameters": {"m": 16, "ef_construction": 512}}}}}})
    kept = []
    def gen():
        for did in doc_mv:
            mv = doc_mv[did]; n = len(mv); k = max(1, int(round(n * R / 100.0)))
            keep = doc_order[did][:k]; kept.append(k)
            yield {"_index": IDX, "_id": did, "muvera_fde": doc_fde[did],
                   "colbert_vectors": quant(mv[keep].tolist(), variant)}
    t = time.time(); helpers.bulk(c, gen(), chunk_size=20, request_timeout=1800, raise_on_error=False)
    c.indices.refresh(index=IDX)
    scalars_kept = int(np.sum(kept)) * DIM
    true_mv_mb = round(scalars_kept * BYTES_PER[variant] / 1024 / 1024, 1)
    avg_kept = float(np.mean(kept))
    log(f"[{variant}_{R}] indexed {c.count(index=IDX)['count']} in {time.time()-t:.0f}s avg_kept={avg_kept:.0f} true_mv_mb={true_mv_mb}")
    for qid in list(queries)[:5]:
        c.transport.perform_request("POST", f"/{IDX}/_search", body=body(q_fde[qid], queries[qid]["embeddings"]))
    took = []; r1 = []; r10 = []; mrrs = []; ndcgs = []
    for qid, q in queries.items():
        rel = qrels.get(qid)
        if not rel: continue
        r = c.transport.perform_request("POST", f"/{IDX}/_search", body=body(q_fde[qid], q["embeddings"]))
        took.append(r["took"]); ranked = [h["_id"] for h in r["hits"]["hits"]]
        r1.append(recall(ranked, rel, 1)); r10.append(recall(ranked, rel, 10))
        mrrs.append(mrr(ranked, rel, 10)); ndcgs.append(ndcg(ranked, rel, 10))
    res = {"variant": variant, "retain_pct": R, "avg_doc_tokens_kept": round(avg_kept, 1),
           "ndcg@10": round(float(np.mean(ndcgs)), 4), "mrr@10": round(float(np.mean(mrrs)), 4),
           "recall@1": round(float(np.mean(r1)), 4), "recall@10": round(float(np.mean(r10)), 4),
           "lat_avg_ms": round(float(np.mean(took)), 1), "lat_p95_ms": round(float(np.percentile(took, 95)), 1),
           "true_mv_mb": true_mv_mb, "n": len(ndcgs)}
    results["cells"][f"{variant}_{R}"] = res; log(f"[{variant}_{R}] {res}")
    c.indices.delete(index=IDX)
json.dump(results, open("/home/ec2-user/quant_prune_irpapers_proper.json", "w"), indent=2)
log("SAVED"); log(json.dumps(results, indent=2))
