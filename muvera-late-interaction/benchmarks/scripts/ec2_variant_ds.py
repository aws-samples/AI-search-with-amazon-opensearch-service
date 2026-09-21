"""One MUVERA variant on a chosen dataset (standalone-plugin 3.8.0, port 9210).
Measures MUVERA+rerank and FDE-only nDCG@1/5/10 + latency. No deletes.
Usage: ec2_variant_ds.py <data_dir> <name> <k_sim> <dim_proj> <r_reps>
"""
import json, math, sys, time
import numpy as np
from opensearchpy import OpenSearch, helpers
def log(*a): print(*a); sys.stdout.flush()

DATA, NAME, K_SIM, DIM_PROJ, R_REPS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
DIM, SEED = 128, 42
FDE = R_REPS*(1<<K_SIM)*DIM_PROJ
SIZE, OVERSAMPLE, SPACE = 10, 4, "innerproduct"
IDX=f"muvera-{NAME}"; ING=f"mving-{NAME}"; SR=f"mvsrch-{NAME}"
c=OpenSearch(hosts=[{"host":"127.0.0.1","port":9210}],use_ssl=False,timeout=600)
log(f"VARIANT {NAME} data={DATA}: k_sim={K_SIM} dim_proj={DIM_PROJ} r_reps={R_REPS} -> FDE={FDE}")
assert FDE<=16000

docs=json.load(open(f"{DATA}/doc_embeddings.json"))
queries=json.load(open(f"{DATA}/query_embeddings.json"))
qrels=json.load(open(f"{DATA}/qrels.json"))
log(f"docs={len(docs)} q={len(queries)} qrels={len(qrels)}")

pp={"dim":DIM,"k_sim":K_SIM,"dim_proj":DIM_PROJ,"r_reps":R_REPS,"seed":SEED,"fde_dimension":FDE}
c.ingest.put_pipeline(id=ING,body={"processors":[{"muvera":{**pp,"source_field":"colbert_vectors","target_field":"muvera_fde"}}]})
c.transport.perform_request("PUT",f"/_search/pipeline/{SR}",body={"request_processors":[{"muvera_query":{**pp,"target_field":"muvera_fde"}}]})
if not c.indices.exists(index=IDX):
    c.indices.create(index=IDX,body={"settings":{"index.knn":True,"default_pipeline":ING,"number_of_shards":1,"number_of_replicas":0},
        "mappings":{"properties":{"colbert_vectors":{"type":"object","enabled":False},
            "muvera_fde":{"type":"knn_vector","dimension":FDE,"method":{"name":"hnsw","engine":"lucene","space_type":SPACE,"parameters":{"m":16,"ef_construction":512}}}}}})
    t=time.time()
    def gen():
        for did,d in docs.items(): yield {"_index":IDX,"_id":did,"colbert_vectors":d["embeddings"]}
    ok,errs=helpers.bulk(c,gen(),chunk_size=100,request_timeout=600,raise_on_error=False)
    c.indices.refresh(index=IDX)
    log(f"indexed ok={ok} errs={len(errs) if isinstance(errs,list) else errs} in {time.time()-t:.0f}s")
log("count:",c.count(index=IDX)["count"])

def dcg(r): return sum((2**x-1)/math.log2(i+2) for i,x in enumerate(r))
def ndcg(ranked,rel,k):
    idcg=dcg(sorted(rel.values(),reverse=True)[:k]); return dcg([rel.get(d,0) for d in ranked[:k]])/idcg if idcg>0 else 0.0
def body(qv,rerank):
    k=SIZE*OVERSAMPLE if rerank else SIZE
    src="lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)" if rerank else "_score"
    t={"script_score":{"query":{"knn":{"muvera_fde":{"vector":"${muvera_fde}","k":k}}},"script":{"source":src,"params":{"query_vectors":qv,"space_type":SPACE}}}}
    return {"query":{"template":t},"size":SIZE,"_source":False}
def evalq(rerank):
    acc={1:[],5:[],10:[]}; took=[]
    for qid,q in queries.items():
        rel=qrels.get(qid)
        if not rel: continue
        r=c.transport.perform_request("POST",f"/{IDX}/_search?search_pipeline={SR}",body=body(q["embeddings"],rerank))
        took.append(r["took"]); ranked=[h["_id"] for h in r["hits"]["hits"]]
        for k in (1,5,10): acc[k].append(ndcg(ranked,rel,k))
    m={f"ndcg@{k}":round(float(np.mean(v)),4) for k,v in acc.items()}
    m["lat_avg_ms"]=round(float(np.mean(took)),1); m["lat_p95_ms"]=round(float(np.percentile(took,95)),1); m["n"]=len(acc[10])
    return m
for qid in list(queries)[:10]:
    c.transport.perform_request("POST",f"/{IDX}/_search?search_pipeline={SR}",body=body(queries[qid]["embeddings"],True))
res={"variant":NAME,"data":DATA.split("/")[-1],"k_sim":K_SIM,"dim_proj":DIM_PROJ,"r_reps":R_REPS,"fde":FDE}
log("MUVERA+rerank..."); res["muvera_rerank"]=evalq(True); log("  ",res["muvera_rerank"])
log("FDE-only..."); res["fde_only"]=evalq(False); log("  ",res["fde_only"])
json.dump(res,open(f"/home/ec2-user/sf_{NAME}.json","w"),indent=2)
log("SAVED"); log(json.dumps(res,indent=2))
