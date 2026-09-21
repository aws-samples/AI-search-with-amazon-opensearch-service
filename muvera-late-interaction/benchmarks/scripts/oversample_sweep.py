"""
Rescore-depth (oversample) sweep on an EXISTING plugin-built MUVERA index (port 9210).
Reuses the plugin's muvera_query search processor (server-side FDE), varies ONLY oversample.
Does NOT reindex, does NOT delete. Measures nDCG@1/5/10 + latency at each oversample.

Usage: oversample_sweep.py <data_dir> <index> <k_sim> <dim_proj> <r_reps> <oversamples csv>
  e.g. oversample_sweep.py data muvera-nfcorpus 5 16 20 4,2
"""
import json, math, sys, time
import numpy as np
from opensearchpy import OpenSearch

def log(*a): print(*a); sys.stdout.flush()

DATA, IDX = sys.argv[1], sys.argv[2]
K_SIM, DIM_PROJ, R_REPS = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
OVERS = [int(x) for x in sys.argv[6].split(",")]
DIM, SEED = 128, 42
FDE = R_REPS * (1 << K_SIM) * DIM_PROJ
SIZE, SPACE = 10, "innerproduct"
SR = f"mvsweep-{IDX}"
c = OpenSearch(hosts=[{"host": "127.0.0.1", "port": 9210}], use_ssl=False, timeout=600)
log(f"SWEEP idx={IDX} FDE={FDE} k_sim={K_SIM} dim_proj={DIM_PROJ} r_reps={R_REPS} oversamples={OVERS}")

queries = json.load(open(f"{DATA}/query_embeddings.json"))
qrels = json.load(open(f"{DATA}/qrels.json"))
log(f"queries={len(queries)} qrels={len(qrels)} indexed_docs={c.count(index=IDX)['count']}")

pp = {"dim": DIM, "k_sim": K_SIM, "dim_proj": DIM_PROJ, "r_reps": R_REPS, "seed": SEED,
      "fde_dimension": FDE, "target_field": "muvera_fde"}
c.transport.perform_request("PUT", f"/_search/pipeline/{SR}",
                            body={"request_processors": [{"muvera_query": pp}]})

def dcg(r): return sum((2**x - 1) / math.log2(i + 2) for i, x in enumerate(r))
def ndcg(ranked, rel, k):
    idcg = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(d, 0) for d in ranked[:k]]) / idcg if idcg > 0 else 0.0

def body(qv, oversample):
    k = SIZE * oversample
    src = "lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)"
    t = {"script_score": {"query": {"knn": {"muvera_fde": {"vector": "${muvera_fde}", "k": k}}},
         "script": {"source": src, "params": {"query_vectors": qv, "space_type": SPACE}}}}
    # Exclude all vector fields from the response _source so no vector payload is serialized
    # back to the client (reduces response size / fetch cost). Scoring still reads vectors from
    # the shard-side _source internally; this only affects what is returned in hits.
    return {"query": {"template": t}, "size": SIZE,
            "_source": {"excludes": ["colbert_vectors", "muvera_fde"]}}

def warmup_index():
    # Load the FDE HNSW graph into memory for THIS index only (one index at a time).
    c.transport.perform_request("GET", f"/_plugins/_knn/warmup/{IDX}")
    # Prime the query path (encode + prefetch + rerank) with a handful of real queries.
    for qid in list(queries)[:10]:
        c.transport.perform_request("POST", f"/{IDX}/_search?search_pipeline={SR}",
                                    body=body(queries[qid]["embeddings"], OVERS[0]))

def evalq(oversample):
    acc = {1: [], 5: [], 10: []}; took = []
    for qid, q in queries.items():
        rel = qrels.get(qid)
        if not rel: continue
        r = c.transport.perform_request("POST", f"/{IDX}/_search?search_pipeline={SR}",
                                        body=body(q["embeddings"], oversample))
        took.append(r["took"]); ranked = [h["_id"] for h in r["hits"]["hits"]]
        for k in (1, 5, 10): acc[k].append(ndcg(ranked, rel, k))
    m = {f"ndcg@{k}": round(float(np.mean(v)), 4) for k, v in acc.items()}
    m["lat_avg_ms"] = round(float(np.mean(took)), 1)
    m["lat_p95_ms"] = round(float(np.percentile(took, 95)), 1)
    m["rescore_depth"] = SIZE * oversample; m["n"] = len(acc[10])
    return m

res = {"index": IDX, "data": DATA.split("/")[-1], "fde": FDE, "by_oversample": {}}
# record segment count for this index
try:
    seg = c.transport.perform_request("GET", f"/_cat/segments/{IDX}?h=segment&format=json")
    res["num_segments"] = len(seg)
    log("num_segments:", res["num_segments"])
except Exception as e:
    log("seg count failed:", e)
log("warming up FDE graph + query path for", IDX, "...")
warmup_index()
for ov in OVERS:
    log(f"oversample={ov} (rescore depth {SIZE*ov})...")
    m = evalq(ov); res["by_oversample"][str(ov)] = m
    log("   ", m)
out = f"/home/ec2-user/oversample_{IDX}.json"
json.dump(res, open(out, "w"), indent=2)
log("SAVED", out); log(json.dumps(res, indent=2))
