"""
COMBINED quantization x norm-pruning on the SAME SciFact index/pipeline as the quant slide.

Matches quant_multivector_test.py EXACTLY so the float32/100% cell reproduces the slide's
baseline (nDCG@10 = 0.6682 @ ~365 ms):
  - embeddings: original data_scifact/doc_embeddings.json ["embeddings"] (full ~235 tok)
  - client FDE: K_SIM=4, DIM_PROJ=16, R_REPS=40 -> FDE=10240, SEED=42
  - SIZE=10, OVERSAMPLE=4 -> 40 rerank candidates; rerank reads colbert_vectors via
    lateInteractionScore; _source excluded otherwise
  - int8: symmetric per-vector quantize->dequantize round-trip (stored dequantized in _source)

Crosses {float32, int8} x retention {100,75,50}%. Norm-based doc pruning (L2 magnitude),
keep top-R% tokens per doc. FULL-token float32 FDE used for phase-1 prefetch in every cell,
so the candidate set is identical -> isolates rerank effect of (precision x token count).

Reports per cell: nDCG@10, rerank lat avg/p95, avg doc tokens kept, and TRUE COMPACT multivector
MB = bytes_per(variant) * scalars_kept_after_prune (the storage story; NOT _source size).

Run on EC2 in ~/k-NN/benchmarks/muvera with the 3.8 cluster (9210) up:
    python3 quant_prune_combined.py <max_docs|0=all>
Writes /home/ec2-user/quant_prune_combined_scifact.json
"""
import json, math, sys, time
import numpy as np
from opensearchpy import OpenSearch, helpers

DATA = sys.argv[2] if len(sys.argv) > 2 else "data_scifact"
MAXD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DIM, SEED = 128, 42
K_SIM, DIM_PROJ, R_REPS = 4, 16, 40
FDE = R_REPS * (1 << K_SIM) * DIM_PROJ
SIZE, OVERSAMPLE, SPACE = 10, 4, "innerproduct"
PORT = 9210
RETAIN = [100, 50]
VARIANTS = ["float32", "int8"]
BYTES_PER = {"float32": 4, "fp16": 2, "int8": 1}
c = OpenSearch(hosts=[{"host": "127.0.0.1", "port": PORT}], use_ssl=False, timeout=600)
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
def quant(mv, variant):
    return mv if variant == "float32" else q_int8(mv)

def dcg(r): return sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(r))
def ndcg(ranked, rel, k):
    idcg = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(d, 0) for d in ranked[:k]]) / idcg if idcg > 0 else 0.0

docs = json.load(open(f"{DATA}/doc_embeddings.json"))
queries = json.load(open(f"{DATA}/query_embeddings.json"))
qrels = json.load(open(f"{DATA}/qrels.json"))
if MAXD: docs = dict(list(docs.items())[:MAXD])
log(f"docs={len(docs)} q={len(queries)} FDE={FDE} variants={VARIANTS} retain={RETAIN}")

enc = Fde(DIM, K_SIM, DIM_PROJ, R_REPS, SEED)
doc_mv = {}; doc_fde = {}; doc_order = {}
t = time.time()
for did, d in docs.items():
    mv = np.asarray(d["embeddings"], np.float32)
    doc_mv[did] = mv
    doc_fde[did] = enc.encode(mv, True).tolist()          # full-token float32 FDE for prefetch
    doc_order[did] = np.argsort(-np.linalg.norm(mv, axis=1))  # high-norm first
q_fde = {qid: enc.encode(q["embeddings"], False).tolist() for qid, q in queries.items()}
log(f"client FDE precompute {time.time()-t:.0f}s")

src = "lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)"
def body(qfde, qmv):
    return {"query": {"script_score": {"query": {"knn": {"muvera_fde": {"vector": qfde, "k": SIZE * OVERSAMPLE}}},
            "script": {"source": src, "params": {"query_vectors": qmv, "space_type": SPACE}}}},
            "size": SIZE, "_source": {"excludes": ["colbert_vectors", "muvera_fde"]}}

results = {"data": DATA, "fde": FDE, "num_docs": len(docs), "cells": {}}
for variant in VARIANTS:
    for R in RETAIN:
        cell = f"{variant}_{R}"
        IDX = f"muvera-qp-{variant}-{R}"
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
                pruned = mv[keep]
                yield {"_index": IDX, "_id": did, "muvera_fde": doc_fde[did],
                       "colbert_vectors": quant(pruned.tolist(), variant)}
        t = time.time(); helpers.bulk(c, gen(), chunk_size=50, request_timeout=600, raise_on_error=False)
        c.indices.refresh(index=IDX)
        scalars_kept = int(np.sum(kept)) * DIM
        true_mv_mb = round(scalars_kept * BYTES_PER[variant] / 1024 / 1024, 1)
        avg_kept = float(np.mean(kept))
        log(f"[{cell}] indexed {c.count(index=IDX)['count']} in {time.time()-t:.0f}s avg_kept={avg_kept:.0f} true_mv_mb={true_mv_mb}")
        for qid in list(queries)[:10]:
            c.transport.perform_request("POST", f"/{IDX}/_search", body=body(q_fde[qid], queries[qid]["embeddings"]))
        took = []; acc = []
        for qid, q in queries.items():
            rel = qrels.get(qid)
            if not rel: continue
            r = c.transport.perform_request("POST", f"/{IDX}/_search", body=body(q_fde[qid], q["embeddings"]))
            took.append(r["took"]); acc.append(ndcg([h["_id"] for h in r["hits"]["hits"]], rel, 10))
        res = {"variant": variant, "retain_pct": R, "avg_doc_tokens_kept": round(avg_kept, 1),
               "ndcg@10": round(float(np.mean(acc)), 4),
               "lat_avg_ms": round(float(np.mean(took)), 1), "lat_p95_ms": round(float(np.percentile(took, 95)), 1),
               "true_mv_mb": true_mv_mb, "n": len(acc)}
        results["cells"][cell] = res; log(f"[{cell}] {res}")
        c.indices.delete(index=IDX)   # free space between cells (one cluster, limited disk)
json.dump(results, open(f"/home/ec2-user/quant_prune_combined_{DATA}.json", "w"), indent=2)
log("SAVED"); log(json.dumps(results, indent=2))
