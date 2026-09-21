"""IRPAPERS k_sim experiment on standalone plugin (3.8.0, port 9210), ef_search=1024.
Per variant: index (if absent) via our muvera processor, then measure
  - FDE-only recall@1/5/10/20 + latency (ef_search=1024)
  - if --rerank: FDE+rerank recall + latency across rescore-depth k list
Usage: ir_ksim.py <name> <k_sim> <dim_proj> <r_reps> <mode:fdeonly|rerank> [rescore_ks_csv]
"""
import json, sys, time
import numpy as np
from opensearchpy import OpenSearch, helpers
def log(*a): print(*a); sys.stdout.flush()

NAME,K_SIM,DIM_PROJ,R_REPS,MODE = sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
RESCORE_KS = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv)>6 else [1024]
DIM,SEED,EF=128,42,1024
FDE=R_REPS*(1<<K_SIM)*DIM_PROJ
SIZE=20; SPACE="innerproduct"
IDX=f"muvera-irk-{NAME}"; ING=f"irk-ing-{NAME}"; SR=f"irk-srch-{NAME}"
D="/home/ec2-user/k-NN/benchmarks/muvera/data_irpapers"
c=OpenSearch(hosts=[{"host":"127.0.0.1","port":9210}],use_ssl=False,timeout=1800)
log(f"IRK {NAME}: k_sim={K_SIM} dim_proj={DIM_PROJ} r_reps={R_REPS} -> FDE={FDE}, ef_search={EF}, mode={MODE}")
assert FDE<=16000

queries=json.load(open(f"{D}/query_embeddings.json"))
qrels=json.load(open(f"{D}/qrels.json"))

pp={"dim":DIM,"k_sim":K_SIM,"dim_proj":DIM_PROJ,"r_reps":R_REPS,"seed":SEED,"fde_dimension":FDE}
c.ingest.put_pipeline(id=ING,body={"processors":[{"muvera":{**pp,"source_field":"colbert_vectors","target_field":"muvera_fde"}}]})
c.transport.perform_request("PUT",f"/_search/pipeline/{SR}",body={"request_processors":[{"muvera_query":{**pp,"target_field":"muvera_fde"}}]})
if not c.indices.exists(index=IDX):
    c.indices.create(index=IDX,body={"settings":{"index.knn":True,"default_pipeline":ING,"number_of_shards":1,"number_of_replicas":0,"refresh_interval":"-1"},
        "mappings":{"properties":{"colbert_vectors":{"type":"object","enabled":False},
            "muvera_fde":{"type":"knn_vector","dimension":FDE,"method":{"name":"hnsw","engine":"lucene","space_type":SPACE,"parameters":{"m":16,"ef_construction":512}}}}}})
    log("loading 7.7GB docs..."); t=time.time(); docs=json.load(open(f"{D}/doc_embeddings.json"))
    log(f"loaded {len(docs)} in {time.time()-t:.0f}s; indexing..."); t=time.time()
    def gen():
        for did,d in docs.items(): yield {"_index":IDX,"_id":did,"colbert_vectors":d["embeddings"]}
    ok,errs=helpers.bulk(c,gen(),chunk_size=20,request_timeout=1800,raise_on_error=False)
    c.indices.refresh(index=IDX); del docs
    log(f"indexed ok={ok} errs={len(errs) if isinstance(errs,list) else errs} in {time.time()-t:.0f}s")
log("count:",c.count(index=IDX)["count"])

def recall(ranked,rel,k):
    rs=set(rel.keys()); return len(rs & set(ranked[:k]))/len(rs) if rs else 0.0
def knn_clause(k):
    return {"knn":{"muvera_fde":{"vector":"${muvera_fde}","k":k,"method_parameters":{"ef_search":EF}}}}
def body(qv,rerank,rescore_k):
    if rerank:
        src="lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)"
        t={"script_score":{"query":knn_clause(rescore_k),"script":{"source":src,"params":{"query_vectors":qv,"space_type":SPACE}}}}
    else:
        t={"script_score":{"query":knn_clause(SIZE),"script":{"source":"_score","params":{"query_vectors":qv,"space_type":SPACE}}}}
    return {"query":{"template":t},"size":SIZE,"_source":False}
def evalq(rerank,rescore_k):
    acc={1:[],5:[],10:[],20:[]}; took=[]
    for qid,q in queries.items():
        rel=qrels.get(qid)
        if not rel: continue
        r=c.transport.perform_request("POST",f"/{IDX}/_search?search_pipeline={SR}",body=body(q["embeddings"],rerank,rescore_k))
        took.append(r["took"]); ranked=[h["_id"] for h in r["hits"]["hits"]]
        for k in (1,5,10,20): acc[k].append(recall(ranked,rel,k))
    m={f"recall@{k}":round(float(np.mean(v)),4) for k,v in acc.items()}
    m["lat_avg_ms"]=round(float(np.mean(took)),1); m["lat_p95_ms"]=round(float(np.percentile(took,95)),1)
    return m
# warmup
for qid in list(queries)[:5]:
    c.transport.perform_request("POST",f"/{IDX}/_search?search_pipeline={SR}",body=body(queries[qid]["embeddings"],False,SIZE))

res={"variant":NAME,"k_sim":K_SIM,"dim_proj":DIM_PROJ,"r_reps":R_REPS,"fde":FDE,"ef_search":EF}
log("FDE-only..."); res["fde_only"]=evalq(False,SIZE); log("  ",res["fde_only"])
if MODE=="rerank":
    res["rerank"]={}
    for rk in RESCORE_KS:
        log(f"rerank rescore_k={rk} ..."); m=evalq(True,rk); res["rerank"][str(rk)]=m; log("  ",m)
json.dump(res,open(f"/home/ec2-user/irk_{NAME}.json","w"),indent=2)
log("SAVED irk_"+NAME+".json"); log(json.dumps(res,indent=2))
