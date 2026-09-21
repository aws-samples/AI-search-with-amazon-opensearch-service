"""Latency measurement for MUVERA in OUR standalone-plugin mode (server-side FDE encode).
Runs ON the EC2 box against the live muvera-nfcorpus index on port 9210.
Reports server 'took' (pure OpenSearch time) and client end-to-end wall time.
Configs: MUVERA+rerank(4x) and MUVERA FDE-only (no rerank)."""
import json, time, sys
import numpy as np
from opensearchpy import OpenSearch

def log(*a): print(*a); sys.stdout.flush()
PORT=9210; INDEX="muvera-nfcorpus"; DATA="/home/ec2-user/k-NN/benchmarks/muvera/data"
SIZE, OVERSAMPLE, SPACE = 10, 4, "innerproduct"
c=OpenSearch(hosts=[{"host":"127.0.0.1","port":PORT}],use_ssl=False,timeout=120)

queries=json.load(open(f"{DATA}/query_embeddings.json"))
qids=list(queries.keys())
log(f"{len(qids)} queries loaded")

def make_body(qv, rerank):
    k = SIZE*OVERSAMPLE if rerank else SIZE
    src = "lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)" if rerank else "1.0"
    tmpl={"script_score":{"query":{"knn":{"muvera_fde":{"vector":"${muvera_fde}","k":k}}},
        "script":{"source":src,"params":{"query_vectors":qv,"space_type":SPACE}}}}
    return {"query":{"template":tmpl},"size":SIZE,"_source":False}

def bench(rerank, name, warmup=20, iters=None):
    took=[]; wall=[]
    ids = qids if iters is None else qids[:iters]
    # warmup
    for qid in qids[:warmup]:
        c.transport.perform_request("POST",f"/{INDEX}/_search?search_pipeline=mv-srch",body=make_body(queries[qid]["embeddings"],rerank))
    for qid in ids:
        b=make_body(queries[qid]["embeddings"],rerank)
        t0=time.perf_counter()
        r=c.transport.perform_request("POST",f"/{INDEX}/_search?search_pipeline=mv-srch",body=b)
        wall.append((time.perf_counter()-t0)*1000)
        took.append(r["took"])
    def pct(a,p): return round(float(np.percentile(a,p)),1)
    m={"n":len(ids),
       "server_took_ms":{"avg":round(float(np.mean(took)),1),"p50":pct(took,50),"p95":pct(took,95),"p99":pct(took,99)},
       "client_wall_ms":{"avg":round(float(np.mean(wall)),1),"p50":pct(wall,50),"p95":pct(wall,95),"p99":pct(wall,99)}}
    log(f"== {name} =="); log(json.dumps(m,indent=2))
    return m

res={}
res["MUVERA + MaxSim rerank (4x)"]=bench(True,"MUVERA + rerank (oversample=4)")
res["MUVERA FDE-only (no rerank)"]=bench(False,"MUVERA FDE-only")
json.dump(res,open("/home/ec2-user/ec2_latency_results.json","w"),indent=2)
log("\nSAVED ec2_latency_results.json")
