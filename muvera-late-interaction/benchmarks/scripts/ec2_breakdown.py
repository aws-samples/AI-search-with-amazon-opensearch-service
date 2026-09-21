import json, statistics, sys
from opensearchpy import OpenSearch
def log(*a): print(*a); sys.stdout.flush()
c=OpenSearch(hosts=[{"host":"127.0.0.1","port":9210}],use_ssl=False,timeout=120)
q=json.load(open("/home/ec2-user/k-NN/benchmarks/muvera/data/query_embeddings.json"))
qids=list(q.keys())

def body(qv, rerank, profile=False):
    k=40 if rerank else 10
    src="lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)" if rerank else "1.0"
    t={"script_score":{"query":{"knn":{"muvera_fde":{"vector":"${muvera_fde}","k":k}}},
       "script":{"source":src,"params":{"query_vectors":qv,"space_type":"innerproduct"}}}}
    b={"query":{"template":t},"size":10,"_source":False}
    if profile: b["profile"]=True
    return b

for qid in qids[:15]:
    c.transport.perform_request("POST","/muvera-nfcorpus/_search?search_pipeline=mv-srch",body=body(q[qid]["embeddings"],True))

rerank=[]; fdeonly=[]
for qid in qids[:40]:
    qv=q[qid]["embeddings"]
    rerank.append(c.transport.perform_request("POST","/muvera-nfcorpus/_search?search_pipeline=mv-srch",body=body(qv,True))["took"])
    fdeonly.append(c.transport.perform_request("POST","/muvera-nfcorpus/_search?search_pipeline=mv-srch",body=body(qv,False))["took"])

med=statistics.median
R=med(rerank); F=med(fdeonly)
log("=== MUVERA+rerank breakdown (median over 40 q, server 'took') ===")
log(f"total MUVERA+rerank : {R} ms")
log(f"  encode+prefetch   : {F} ms   (from FDE-only config)")
log(f"  rerank (MaxSim)    : {R-F} ms  (= total - FDE-only)")

# profile split of query vs rescore inside the shard
prof=c.transport.perform_request("POST","/muvera-nfcorpus/_search?search_pipeline=mv-srch",body=body(q[qids[0]]["embeddings"],True,profile=True))
log("\n=== shard profile (one query) ===")
for s in prof.get("profile",{}).get("shards",[])[:1]:
    for sr in s.get("searches",[]):
        for qn in sr.get("query",[]):
            log("query node:", qn.get("type"), round(qn.get("time_in_nanos",0)/1e6,2),"ms")
        for cc in sr.get("collector",[]):
            log("collector:", cc.get("name"), round(cc.get("time_in_nanos",0)/1e6,2),"ms")
    # rescore phase shows up under 'rescore' if present
json.dump({"total_ms":R,"encode_prefetch_ms":F,"rerank_ms":R-F},open("/home/ec2-user/ec2_breakdown.json","w"),indent=2)
log("\nSAVED")
