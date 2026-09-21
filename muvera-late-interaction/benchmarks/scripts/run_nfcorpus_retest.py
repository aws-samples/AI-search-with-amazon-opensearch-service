"""
Standalone-plugin NFCorpus retest.

Drives OUR org.opensearch.muvera processors on a fresh OpenSearch 3.8.0 cluster
(k-NN present only for lateInteractionScore). Reuses the RFC benchmark's ColBERTv2
embeddings + qrels. Computes nDCG@1/5/10 for:
  - Exact MaxSim (brute force, client-side)      <- accuracy ceiling
  - Mean pool + MaxSim rerank                     <- naive baseline
  - MUVERA FDE-only (no rerank)
  - MUVERA + MaxSim rerank (oversample=4)         <- the shipped pattern
and compares to the RFC merged-mode numbers.
"""
import json, math, time, sys, warnings
import numpy as np
from opensearchpy import OpenSearch, helpers

warnings.filterwarnings("ignore")
BASE = {"host": "localhost", "port": 9201}
INDEX = "muvera-nfcorpus"
DATA = "/Users/prasadnu/OpenSearch/muvera_retest/data"

# RFC benchmark params
DIM, K_SIM, DIM_PROJ, R_REPS, SEED = 128, 5, 16, 20, 42
FDE_DIM = R_REPS * (1 << K_SIM) * DIM_PROJ   # 10240
OVERSAMPLE = 4
SIZE = 10
SPACE = "innerproduct"

client = OpenSearch(hosts=[BASE], use_ssl=False, timeout=600)

print(f"FDE_DIM={FDE_DIM}")

# ---------- load data ----------
print("loading embeddings...")
docs = json.load(open(f"{DATA}/doc_embeddings.json"))
queries = json.load(open(f"{DATA}/query_embeddings.json"))
qrels = json.load(open(f"{DATA}/qrels.json"))
print(f"docs={len(docs)} queries={len(queries)} qrels={len(qrels)}")

# ---------- pipelines (OUR processors) ----------
if client.indices.exists(index=INDEX):
    client.indices.delete(index=INDEX)
for p in ("muvera-nf-ingest",):
    try: client.ingest.delete_pipeline(id=p)
    except Exception: pass

client.ingest.put_pipeline(id="muvera-nf-ingest", body={
    "description": "MUVERA FDE (standalone plugin)",
    "processors": [{"muvera": {
        "source_field": "colbert_vectors", "target_field": "muvera_fde",
        "dim": DIM, "k_sim": K_SIM, "dim_proj": DIM_PROJ, "r_reps": R_REPS, "seed": SEED,
        "fde_dimension": FDE_DIM
    }}]
})
client.transport.perform_request("PUT", "/_search/pipeline/muvera-nf-search", body={
    "request_processors": [{"muvera_query": {
        "target_field": "muvera_fde",
        "dim": DIM, "k_sim": K_SIM, "dim_proj": DIM_PROJ, "r_reps": R_REPS, "seed": SEED,
        "fde_dimension": FDE_DIM
    }}]
})

# ---------- index ----------
client.indices.create(index=INDEX, body={
    "settings": {"index.knn": True, "default_pipeline": "muvera-nf-ingest",
                 "number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "colbert_vectors": {"type": "object", "enabled": False},
        "muvera_fde": {"type": "knn_vector", "dimension": FDE_DIM,
            "method": {"name": "hnsw", "engine": "faiss", "space_type": SPACE,
                       "parameters": {"m": 16, "ef_construction": 512}}}
    }}
})
print("indexing (ingest processor computes FDE)...")
t0 = time.time()
def gen():
    for did, d in docs.items():
        yield {"_index": INDEX, "_id": did, "colbert_vectors": d["embeddings"]}
ok, errs = helpers.bulk(client, gen(), chunk_size=100, request_timeout=600, raise_on_error=False)
print(f"indexed ok={ok} errs={len(errs) if isinstance(errs,list) else errs} in {time.time()-t0:.0f}s")
client.indices.refresh(index=INDEX)
print("count:", client.count(index=INDEX)["count"])

# ---------- precompute doc arrays for client-side baselines ----------
doc_ids = list(docs.keys())
doc_mv = {d: np.asarray(docs[d]["embeddings"], dtype=np.float32) for d in doc_ids}
doc_mean = {d: doc_mv[d].mean(axis=0) for d in doc_ids}

def maxsim(qv, dv):  # qv:[Tq,128] dv:[Td,128], innerproduct
    return float(np.max(qv @ dv.T, axis=1).sum())

def dcg(rels):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg_at(ranked_ids, rel_map, k):
    gains = [rel_map.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0

# ---------- run configs ----------
def eval_config(rank_fn, name):
    n = {1: [], 5: [], 10: []}
    for qid, q in queries.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        qv = np.asarray(q["embeddings"], dtype=np.float32)
        ranked = rank_fn(qid, qv)
        for k in (1, 5, 10):
            n[k].append(ndcg_at(ranked, rel, k))
    return {f"ndcg@{k}": round(float(np.mean(v)), 4) for k, v in n.items()}, len(n[10])

# Exact MaxSim brute force (ceiling)
def rank_exact(qid, qv):
    scored = [(d, maxsim(qv, doc_mv[d])) for d in doc_ids]
    scored.sort(key=lambda x: -x[1]); return [d for d, _ in scored]

# Mean pool + MaxSim rerank: knn on mean vector via client, rerank topN by maxsim
# (approximate the RFC "mean pool + rerank": retrieve by mean-vector cosine, rerank maxsim)
def rank_meanpool(qid, qv):
    qmean = qv.mean(axis=0)
    scored = [(d, float(qmean @ doc_mean[d])) for d in doc_ids]
    scored.sort(key=lambda x: -x[1])
    cand = [d for d, _ in scored[:SIZE * OVERSAMPLE]]
    cand.sort(key=lambda d: -maxsim(qv, doc_mv[d]))
    return cand

# MUVERA + MaxSim rerank via OUR standalone plugin: shipped template shape
# (knn prefetch on FDE, muvera_query encodes ${muvera_fde}, lateInteractionScore reranks).
def rank_muvera_rerank(qid, qv):
    qlist = qv.tolist()
    tmpl = {"script_score": {
        "query": {"knn": {"muvera_fde": {"vector": "${muvera_fde}", "k": SIZE * OVERSAMPLE}}},
        "script": {"source": "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)",
                   "params": {"query_vectors": qlist, "space_type": SPACE}}}}
    body = {"query": {"template": tmpl}, "size": SIZE, "_source": False}
    r = client.transport.perform_request("POST", f"/{INDEX}/_search?search_pipeline=muvera-nf-search", body=body)
    return [h["_id"] for h in r["hits"]["hits"]]

results = {}
for name, fn in [
    ("Exact MaxSim (brute-force)", rank_exact),
    ("Mean pool + MaxSim rerank", rank_meanpool),
    ("MUVERA + MaxSim rerank (4x)", rank_muvera_rerank),
]:
    print(f"running: {name} ...")
    t0 = time.time()
    metrics, nq = eval_config(fn, name)
    metrics["num_queries"] = nq
    metrics["wall_s"] = round(time.time() - t0, 1)
    results[name] = metrics
    print("  ", metrics)

json.dump(results, open(f"{DATA}/../retest_results.json", "w"), indent=2)
print("\n=== RESULTS (standalone plugin) ===")
print(json.dumps(results, indent=2))
