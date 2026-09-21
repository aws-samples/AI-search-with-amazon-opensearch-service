"""
Client-side multivector quantization test (SciFact) on standalone plugin 3.8.0 (port 9210).

Design:
- FDE is generated CLIENT-SIDE (numpy, self-consistent random matrices; NOT the plugin processor).
  The SAME float32 FDE is used for prefetch in ALL variants, so prefetch candidates are identical
  across variants -> isolates the effect of MULTIVECTOR quantization on the RERANK step only.
- We vary only the stored `colbert_vectors` precision:
    float32 : raw floats (baseline, what we store today)
    fp16    : values round-tripped through float16 (IEEE half) then back to float (precision loss only)
    int8    : symmetric per-doc int8 quantize->dequantize round-trip (ColBERT vecs ~ L2-normalized)
  lateInteractionScore reads colbert_vectors from _source as numbers, so we store the DEQUANTIZED
  values (round-trip). This measures the RECALL/nDCG impact of the precision loss, and the _source
  size difference is measured separately from the TRUE compact byte form.

Measures per variant: nDCG@1/5/10, rerank latency (avg/p95), FDE-only nDCG (sanity, identical across
variants), index store size, and computed true-compact multivector byte size.

Usage: quant_multivector_test.py <data_dir> <max_docs|0=all>
One cluster at a time; does NOT delete any existing index.
"""
import json, math, sys, time, os
import numpy as np
from opensearchpy import OpenSearch, helpers

def log(*a): print(*a); sys.stdout.flush()

DATA = sys.argv[1]
MAXD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DIM, SEED = 128, 42
K_SIM, DIM_PROJ, R_REPS = 4, 16, 40          # -> FDE 10240, matches muvera-sf_base
FDE = R_REPS * (1 << K_SIM) * DIM_PROJ
NUM_PART = 1 << K_SIM
SIZE, OVERSAMPLE, SPACE = 10, 4, "innerproduct"
PORT = 9210
c = OpenSearch(hosts=[{"host": "127.0.0.1", "port": PORT}], use_ssl=False, timeout=600)
assert FDE <= 16000
log(f"QUANT TEST data={DATA} maxdocs={MAXD or 'all'} FDE={FDE} (k_sim={K_SIM} dim_proj={DIM_PROJ} r_reps={R_REPS})")

# ---------------- client-side FDE encoder (self-consistent; not plugin-compatible) ----------------
class FdeEncoder:
    def __init__(self, dim, k_sim, dim_proj, r_reps, seed):
        self.dim, self.k, self.dp, self.r = dim, k_sim, dim_proj, r_reps
        self.parts = 1 << k_sim
        rng = np.random.RandomState(seed)
        # SimHash hyperplanes [r][k][dim], gaussian
        self.planes = rng.randn(r_reps, k_sim, dim).astype(np.float32)
        # projection rows [r][dim_proj][dim] in {-1,+1}
        self.proj = np.where(rng.rand(r_reps, dim_proj, dim) < 0.5, -1.0, 1.0).astype(np.float32)
        self.scale = 1.0 / math.sqrt(dim_proj)

    def _cluster_ids(self, vecs, r):
        # vecs [n,dim] -> [n] cluster id via sign of dot with each of k planes
        dots = vecs @ self.planes[r].T          # [n,k]
        bits = (dots > 0).astype(np.int32)      # [n,k]
        weights = (1 << np.arange(self.k)).astype(np.int32)
        return bits @ weights                    # [n]

    def encode(self, vecs, is_doc):
        vecs = np.asarray(vecs, dtype=np.float32)
        n = vecs.shape[0]
        out = np.zeros(self.r * self.parts * self.dp, dtype=np.float32)
        off = 0
        for r in range(self.r):
            cids = self._cluster_ids(vecs, r)
            centers = np.zeros((self.parts, self.dim), dtype=np.float32)
            counts = np.zeros(self.parts, dtype=np.int32)
            for v in range(n):
                centers[cids[v]] += vecs[v]
                counts[cids[v]] += 1
            if is_doc:
                nz = counts > 1
                centers[nz] /= counts[nz][:, None]
                # fill empty clusters from hamming-nearest non-empty
                empties = np.where(counts == 0)[0]
                nonempty = np.where(counts > 0)[0]
                if len(empties) and len(nonempty):
                    for cpart in empties:
                        d = np.array([bin(int(cpart) ^ int(o)).count("1") for o in nonempty])
                        nearest = nonempty[int(np.argmin(d))]
                        # first vec assigned to nearest
                        idx = int(np.where(cids == nearest)[0][0])
                        centers[cpart] = vecs[idx]
            # random projection
            projected = (centers @ self.proj[r].T) * self.scale   # [parts, dim_proj]
            block = projected.reshape(-1)
            out[off:off + block.shape[0]] = block
            off += block.shape[0]
        return out

enc = FdeEncoder(DIM, K_SIM, DIM_PROJ, R_REPS, SEED)

# ---------------- quantizers (round-trip: quantize then dequantize back to float) --------------
def q_float32(mv):
    return mv  # passthrough

def q_fp16(mv):
    # round-trip through IEEE half precision
    return [np.asarray(v, dtype=np.float16).astype(np.float32).tolist() for v in mv]

def q_int8(mv):
    # symmetric per-vector int8 (ColBERT vecs L2-normalized -> values in [-1,1] typically)
    out = []
    for v in mv:
        a = np.asarray(v, dtype=np.float32)
        amax = float(np.max(np.abs(a)))
        s = amax / 127.0 if amax > 0 else 1.0
        qi = np.clip(np.round(a / s), -127, 127).astype(np.int8)
        out.append((qi.astype(np.float32) * s).tolist())
    return out

QUANTS = {"float32": q_float32, "fp16": q_fp16, "int8": q_int8}
# true compact bytes-per-scalar for the storage story (not what _source holds)
BYTES_PER = {"float32": 4, "fp16": 2, "int8": 1}

# ---------------- load data ----------------
docs = json.load(open(f"{DATA}/doc_embeddings.json"))
queries = json.load(open(f"{DATA}/query_embeddings.json"))
qrels = json.load(open(f"{DATA}/qrels.json"))
if MAXD:
    docs = dict(list(docs.items())[:MAXD])
log(f"docs={len(docs)} q={len(queries)} qrels={len(qrels)}")

# ---------------- precompute client-side FDE for docs (shared across variants) ----------------
t = time.time()
doc_fde = {}
doc_mv = {}
total_scalars = 0
for did, d in docs.items():
    mv = d["embeddings"]
    doc_mv[did] = mv
    doc_fde[did] = enc.encode(mv, is_doc=True).tolist()
    total_scalars += sum(len(v) for v in mv)
log(f"client FDE for {len(doc_fde)} docs in {time.time()-t:.0f}s; total mv scalars={total_scalars:,}")

# query FDE precompute
q_fde = {qid: enc.encode(q["embeddings"], is_doc=False).tolist() for qid, q in queries.items()}

# ---------------- metrics ----------------
def dcg(r): return sum((2**x - 1) / math.log2(i + 2) for i, x in enumerate(r))
def ndcg(ranked, rel, k):
    idcg = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(d, 0) for d in ranked[:k]]) / idcg if idcg > 0 else 0.0

def search_body(qvec_fde, qmv, rerank):
    k = SIZE * OVERSAMPLE if rerank else SIZE
    if rerank:
        src = "lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)"
        return {"query": {"script_score": {
            "query": {"knn": {"muvera_fde": {"vector": qvec_fde, "k": k}}},
            "script": {"source": src, "params": {"query_vectors": qmv, "space_type": SPACE}}}},
            "size": SIZE, "_source": False}
    else:
        return {"query": {"knn": {"muvera_fde": {"vector": qvec_fde, "k": k}}}, "size": SIZE, "_source": False}

def evalq(idx, rerank):
    acc = {1: [], 5: [], 10: []}; took = []
    for qid, q in queries.items():
        rel = qrels.get(qid)
        if not rel: continue
        r = c.transport.perform_request("POST", f"/{idx}/_search",
                                        body=search_body(q_fde[qid], q["embeddings"], rerank))
        took.append(r["took"]); ranked = [h["_id"] for h in r["hits"]["hits"]]
        for k in (1, 5, 10): acc[k].append(ndcg(ranked, rel, k))
    m = {f"ndcg@{k}": round(float(np.mean(v)), 4) for k, v in acc.items()}
    m["lat_avg_ms"] = round(float(np.mean(took)), 1)
    m["lat_p95_ms"] = round(float(np.percentile(took, 95)), 1)
    m["n"] = len(acc[10])
    return m

# ---------------- per-variant: index + measure ----------------
results = {"data": DATA.split("/")[-1], "fde": FDE, "num_docs": len(docs), "variants": {}}
for vname, qfn in QUANTS.items():
    idx = f"muvera-sfq-{vname}"
    if c.indices.exists(index=idx):
        log(f"[{vname}] index exists, reusing {idx}")
    else:
        c.indices.create(index=idx, body={
            "settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {
                "colbert_vectors": {"type": "object", "enabled": False},
                "muvera_fde": {"type": "knn_vector", "dimension": FDE,
                    "method": {"name": "hnsw", "engine": "lucene", "space_type": SPACE,
                               "parameters": {"m": 16, "ef_construction": 512}}}}}})
        t = time.time()
        def gen():
            for did in docs:
                yield {"_index": idx, "_id": did,
                       "muvera_fde": doc_fde[did],
                       "colbert_vectors": qfn(doc_mv[did])}
        ok, errs = helpers.bulk(c, gen(), chunk_size=50, request_timeout=600, raise_on_error=False)
        c.indices.refresh(index=idx)
        log(f"[{vname}] indexed ok={ok} errs={len(errs) if isinstance(errs,list) else errs} in {time.time()-t:.0f}s")
    # warmup
    for qid in list(queries)[:10]:
        c.transport.perform_request("POST", f"/{idx}/_search",
                                    body=search_body(q_fde[qid], queries[qid]["embeddings"], True))
    v = {}
    log(f"[{vname}] rerank eval...")
    v["muvera_rerank"] = evalq(idx, True); log("   rerank:", v["muvera_rerank"])
    log(f"[{vname}] fde-only eval...")
    v["fde_only"] = evalq(idx, False); log("   fde_only:", v["fde_only"])
    # store size
    st = c.indices.stats(index=idx)["indices"][idx]["primaries"]["store"]["size_in_bytes"]
    v["store_bytes"] = st
    v["store_mb"] = round(st / 1024 / 1024, 1)
    v["true_mv_bytes"] = total_scalars * BYTES_PER[vname]
    v["true_mv_mb"] = round(v["true_mv_bytes"] / 1024 / 1024, 1)
    results["variants"][vname] = v

out = f"/home/ec2-user/quant_mv_{results['data']}.json"
json.dump(results, open(out, "w"), indent=2)
log("SAVED", out)
log(json.dumps(results, indent=2))
