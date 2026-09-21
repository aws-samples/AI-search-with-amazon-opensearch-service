"""nDCG for MUVERA FDE-only (no rerank) in our standalone-plugin mode.
Ranking = knn prefetch order on muvera_fde (query encoded server-side by muvera_query),
NO lateInteractionScore. Compares against qrels."""
import json, math, sys
from opensearchpy import OpenSearch
def log(*a): print(*a); sys.stdout.flush()
c=OpenSearch(hosts=[{"host":"127.0.0.1","port":9210}],use_ssl=False,timeout=120)
DATA="/home/ec2-user/k-NN/benchmarks/muvera/data"
queries=json.load(open(f"{DATA}/query_embeddings.json"))
qrels=json.load(open(f"{DATA}/qrels.json"))

def dcg(r): return sum((2**x-1)/math.log2(i+2) for i,x in enumerate(r))
def ndcg(ranked,rel,k):
    idcg=dcg(sorted(rel.values(),reverse=True)[:k])
    return dcg([rel.get(d,0) for d in ranked[:k]])/idcg if idcg>0 else 0.0

# FDE-only: pure knn on the FDE, query encoded by muvera_query via ${muvera_fde}.
# Use a script_score with constant source so the processor still encodes, but ordering
# is the knn prefetch score (constant script keeps knn order via _score? No -> use plain
# knn with the template so muvera_query fills the vector, size=10, sorted by knn score).
def rank(qv):
    # template with a script_score whose source returns the knn _score so ordering = FDE similarity
    t={"script_score":{"query":{"knn":{"muvera_fde":{"vector":"${muvera_fde}","k":10}}},
       "script":{"source":"_score","params":{"query_vectors":qv,"space_type":"innerproduct"}}}}
    b={"query":{"template":t},"size":10,"_source":False}
    r=c.transport.perform_request("POST","/muvera-nfcorpus/_search?search_pipeline=mv-srch",body=b)
    return [h["_id"] for h in r["hits"]["hits"]]

import numpy as np
acc={1:[],5:[],10:[]}
for qid,q in queries.items():
    rel=qrels.get(qid)
    if not rel: continue
    ranked=rank(q["embeddings"])
    for k in (1,5,10): acc[k].append(ndcg(ranked,rel,k))
m={f"ndcg@{k}":round(float(np.mean(v)),4) for k,v in acc.items()}; m["n"]=len(acc[10])
log("MUVERA FDE-only (no rerank) nDCG:", json.dumps(m))
json.dump(m,open("/home/ec2-user/ec2_fdeonly_ndcg.json","w"))
