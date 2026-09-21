# MUVERA Benchmark Results — Reproducibility Guide

Everything needed to reproduce the numbers in the OpenSearchCon talk
*"Scaling Late Interaction Retrieval in OpenSearch."*

This documents **what we ran, the exact results, and the code to regenerate them** — the
quality comparison (exact / mean-pool / FDE-only / MUVERA+rerank), the latency breakdown, and
the three optimization experiments (rescore depth, FDE compensation, multivector quantization).

> **TL;DR of findings**
> - MUVERA FDE + rerank recovers **90–97% of exact MaxSim**, ~**3× the quality of mean-pooling**.
> - The **rerank is ~97% of query latency** — and it's tunable.
> - **H1 (confirmed):** a higher-quality FDE lets you rerank a shallower candidate pool →
>   **~40% less latency at equal quality**.
> - **H2 (conditional):** FDE-only (skip rerank) is **~35× faster** but only viable for
>   low-density text (~72–76% of exact); dense pages collapse to ~10%.
> - **H3 (rejected for latency):** int8/fp16 quantization is **flat on latency** (cuts bytes,
>   not MaxSim FLOPs) — a memory lever only.

---

## 1. Environment

| | |
|---|---|
| Instance | EC2 g4dn.4xlarge (16 vCPU, 62 GB RAM, single node) |
| OS / JDK | Amazon Linux 2023, JDK 21 |
| OpenSearch | 3.8.0 (self-managed tarball) |
| Plugin | `opensearch-muvera-plugin` (this repo) + stock k-NN plugin |
| k-NN engine | **lucene** (HNSW, `space_type=innerproduct`, `m=16`, `ef_construction=512`) |
| Shards | 1 primary, 0 replica (single-shard; latency is single-shard, parallelizes with more shards) |
| Model | **ColBERTv2** (128-dim tokens) for text; **ColModernVBERT** for IRPAPERS |

> Note on engines: the standalone-plugin retest used **lucene**. An earlier merged-mode run
> used **faiss** (the `*_v3.json` files); expect a ~1–2 pt offset between engines. Quality
> tables below cite the run each number comes from.

---

## 2. Datasets

| Dataset | Docs | Queries | ~vectors/doc | Metric | Modality |
|---|---:|---:|---:|---|---|
| **NFCorpus** | 3,633 | 323 | ~180 | nDCG@10 | text |
| **SciFact** | 5,183 | 300 | ~233 | nDCG@10 | text |
| **IRPAPERS** | 3,230 | 180 | ~1,011 | recall@10 | multimodal (PDF pages) |

Each dataset directory holds three files, keyed by doc/query id:
```
data_<dataset>/
  doc_embeddings.json     # { id: { "text": ..., "embeddings": [[128]xN] } }
  query_embeddings.json   # { id: { "embeddings": [[128]xM] } }
  qrels.json              # { query_id: { doc_id: relevance } }
```

---

## 3. Approaches compared

- **Exact MaxSim (ceiling)** — brute-force MaxSim over the *whole* corpus, client-side. The
  quality ceiling for the model; not deployable.
- **Mean-pool + rerank** — retrieve by a single mean-pooled vector, then MaxSim-rerank the top-k.
- **MUVERA FDE-only** — ANN retrieve on the FDE, no rerank (fast mode).
- **MUVERA + rerank** — ANN retrieve on the FDE, then MaxSim-rerank the top-k (the shipped pattern).

**MaxSim** (the scoring function used by every rerank + the exact ceiling):
```python
def maxsim(qv, dv):            # qv:[Tq,128], dv:[Td,128], inner product
    return float(np.max(qv @ dv.T, axis=1).sum())
```
FDE dimension = `r_reps × 2^k_sim × dim_proj` (cap 16,000). Baseline: `k_sim=5, dim_proj=16,
r_reps=20 → 10,240`. Higher-quality: `r_reps=30 → 15,360`.

---

## 4. Quality — does it work?  (presentation slides 6 & 10)

| Approach | SciFact nDCG@10 | NFCorpus nDCG@10 | IRPAPERS recall@10 |
|---|---:|---:|---:|
| Exact MaxSim (ceiling) | 0.692 | 0.344 | 0.839 |
| **MUVERA + rerank** | **0.671** | **0.311** | **0.733** |
| FDE-only (no rerank) | 0.528 | 0.249 | 0.083 |
| Mean-pool + rerank | 0.363 | 0.144 | 0.233 |

- MUVERA + rerank recovers **97% / 90% / 87%** of exact (SciFact / NFCorpus / IRPAPERS).
- Mean-pool loses **48% / 58% / 72%** of exact — pooling destroys the token-level signal.

*Source: text datasets from `benchmark_results_{scifact,nfcorpus}_v3.json` (faiss run);
IRPAPERS from `benchmark_results_irpapers_v3.json` (MUVERA+rerank = server rerank 4×) and
`ir_meanpool.json` (mean-pool, computed client-side, same pooled-retrieve + MaxSim-rerank method).*

> **Why is exact only 0.344 on NFCorpus?** Exact is the *model's* ceiling, not a perfect oracle.
> NFCorpus is a hard benchmark with many relevant docs per query, so nDCG@10 is structurally low
> for every method — 0.344 matches published ColBERTv2 numbers. The story is the *relative* gap.

### Reproduce
```bash
# text datasets (exact / mean-pool / FDE-only / MUVERA+rerank), nDCG@10
python run_nfcorpus_retest.py            # NFCorpus
python ec2_variant_ds.py data_scifact sf_base 5 16 20   # SciFact MUVERA + FDE-only
# IRPAPERS mean-pool (recall@10), client-side brute force
python ir_meanpool.py
```

---

## 5. Latency — the cost is the rerank  (slide 12)

| Dataset (~vec/doc) | FDE-only | FDE + rerank (4× oversample) |
|---|---:|---:|
| NFCorpus (~180) | 9.7 ms | 340 ms |
| SciFact (~233) | 9.9 ms | 345 ms |
| IRPAPERS (~1,000) | 17 ms | ~1.5 s (depth 40) → 7.5 s (depth 256) |

Breakdown on SciFact: **encode + prefetch ≈ 10 ms, rerank ≈ 307 ms → ~97% of latency is the rerank.**

*Source: `ec2_latency_results.json`, `ec2_breakdown.json`, `irk_ksim4.json`.*

### Reproduce
```bash
python ec2_latency.py       # FDE-only vs FDE+rerank server took (avg/p50/p95/p99)
python ec2_breakdown.py     # splits total into encode+prefetch vs rerank
```

---

## 6. H1 — Better FDE → shallow rerank → lower latency at equal quality  (slides 13–15)

**H1 Part 1 — rescore-depth sweep (baseline FDE, reps20):** latency ~linear in depth.

| Rescore depth | NFCorpus nDCG@10 / lat | SciFact nDCG@10 / lat |
|---|---|---|
| 40 (baseline, 4×) | 0.3036 / 370 ms | 0.6593 / 353 ms |
| 20 (2×) | 0.2812 / 214 ms | 0.6426 / 200 ms |
| 10 (1×) | 0.2598 / 129 ms | 0.6185 / 121 ms |

SciFact 40→20: **−43% latency for −2.5% nDCG.**

**H1 Part 2 — compensation (higher-quality FDE at the shallow depth):**

| Config | SciFact nDCG@10 / lat | NFCorpus nDCG@10 / lat |
|---|---|---|
| reps20 FDE @ depth 40 (baseline) | 0.6593 / 353 ms | 0.3036 / 370 ms |
| **reps30 FDE @ depth 20** | **0.6549 / 216 ms** | **0.3020 / 229 ms** |

→ **Same quality, ~40% less latency.** A better FDE puts the right docs in a smaller shortlist,
so low sampling + FDE tuning are complementary. **Cost:** FDE grows 10,240→15,360 dims
(~+50% FDE memory, ~+8% total shard, +~5 ms prefetch); bounded by the 16,000-dim cap.
**Verdict: confirmed.**

*Source: `oversample_{nfcorpus,scifact}_full.json` (reps20 sweep), `oversample_{sf,nf}_reps30.json`.*

### Reproduce
```bash
# reps20 baseline sweep (depths 40/20/10)
python oversample_sweep.py data_scifact muvera-sf_base   5 16 20 4,2,1
python oversample_sweep.py data        muvera-nfcorpus  5 16 20 4,2,1
# reps30 (higher-quality FDE) at the same depths
python oversample_sweep.py data_scifact muvera-sf_reps30    5 16 30 4,2,1
python oversample_sweep.py data        muvera-nf-v1_reps30 5 16 30 4,2,1
```

---

## 7. H2 — Skip the rerank (FDE-only)?  (slides 16–17)

| Dataset (~vec/doc) | FDE-only quality (% of exact) | Latency |
|---|---:|---:|
| NFCorpus text (~180) | 72% | ~10 ms |
| SciFact text (~233) | 76% | ~10 ms |
| IRPAPERS multimodal (~1,011) | 10% | ~17 ms |

- **~35× faster** (no rerank, ~340 ms → ~10 ms).
- **Quality tracks document density:** text keeps ~72–76% of exact; dense pages collapse to ~10%.
- **Why:** dense docs average ~30 tokens/region (vs ~6 on text) → per-token detail blurs away.
- **Rule of thumb:** the more vectors per document, the more you must keep the rerank.
  **Verdict: conditional** (viable for text, not for dense/PDF-page workloads).

*Source: FDE-only quality from the v3 runs and `sf_base.json` / `ir_tuned_reps30.json`; the
k_sim sweep (`irk_ksim{4,6,7}.json`) confirms higher k_sim does not close the dense-doc gap
under the 16,000-dim cap.*

### Reproduce
```bash
python ec2_fdeonly_ndcg.py                     # FDE-only nDCG (text)
python ir_ksim.py ksim4 4 16 10 fdeonly        # IRPAPERS FDE-only recall (FDE 2560)
python ir_ksim.py ksim6 6 16 10 fdeonly        # k_sim sweep to test the cap
python ir_ksim.py ksim7 7 8  10 fdeonly
```

---

## 8. H3 — Quantize the multivectors?  (slides 18–19)

Client-side test: FDE held identical across variants (isolates the rerank effect); only the
stored `colbert_vectors` precision varies (float32 / fp16 round-trip / int8 symmetric round-trip).

| Variant (SciFact) | nDCG@10 | rerank latency | compact size *(if binary)* |
|---|---:|---:|---:|
| float32 | 0.6682 | 365 ms | 596 MB |
| fp16 | 0.6682 | 370 ms | 298 MB (2×) |
| int8 | 0.6667 | 375 ms | 149 MB (4×) |

- **Quality:** nearly free — int8 −0.2%, fp16 zero (ColBERT vectors are L2-normalized → int8 is clean).
- **Latency:** **flat** — `lateInteractionScore` reads floats from `_source` regardless of
  stored precision; quantization cuts *bytes*, not MaxSim *FLOPs*.
- **Memory:** the 2×/4× is the **potential** compact size — realized only if stored as **binary
  doc-values** (a plugin code change), *not* through `_source` (JSON text doesn't shrink).
  **Verdict: rejected for latency; memory lever only.**

*Source: `quant_mv_scifact.json`.*

### Reproduce
```bash
python quant_multivector_test.py data_scifact 0    # 0 = all docs; builds 3 indices, measures each
```

### The real fix: score off `_source` — native `LateInteractionField` (Lucene 10.5)

The `_source`-bound rerank is exactly what [k-NN RFC #3439](https://github.com/opensearch-project/k-NN/issues/3439)
targets, using **Lucene 10.5's `LateInteractionField`** (multi-vectors as compact
**BinaryDocValues**) + a SIMD `LateInteractionRescorer`. OpenSearch 3.8 already ships Lucene
10.5, so the building blocks are present — it needs a native field type + rescore query (core
work), not the `_source` + Painless path.

We measured the *scoring kernel* directly in Lucene (standalone JVM, not OpenSearch), on SciFact:

| Rerank scoring path (40 candidates × 32 query tokens) | Latency / query |
|---|---:|
| `_source` (deserialize float arrays + manual MaxSim loop) | **440.9 ms** |
| native `LateInteractionField` (binary doc-values + SIMD MaxSim) | **13.1 ms** |
| **Speedup** | **~34×** |

Scores were identical (762.591 vs 762.591) — same MaxSim, faster read+compute path. This
confirms the RFC's 5–20× estimate (we see more because our `_source` path also pays JSON-text
parsing). **Takeaway:** quantization only pays off *once scoring moves off `_source`* — and the
native path is the bigger latency win regardless.

> Caveat: Lucene-kernel microbenchmark, **not** end-to-end OpenSearch latency. It isolates the
> rerank scoring kernel (the ~97%-of-latency component). A true OpenSearch number needs the
> native field type + rescore query (RFC #3439), which is not yet shipped.

### Reproduce (native vs _source kernel)
```bash
# needs Lucene 10.5 jars (bundled with OpenSearch 3.8) on the classpath
CP=$(ls $OPENSEARCH_HOME/lib/lucene-*.jar | tr '\n' ':'):.
javac -cp "$CP" LateInteractionBench.java
java --add-modules jdk.incubator.vector -cp "$CP" \
     LateInteractionBench data_scifact/doc_embeddings.json 2000 40 32 300
```

---

## 9. Scripts index

| Script | What it does |
|---|---|
| `run_nfcorpus_retest.py` | Text: exact / mean-pool / FDE-only / MUVERA+rerank, nDCG@1/5/10 |
| `ec2_variant_ds.py` | One MUVERA variant on a dataset: MUVERA+rerank & FDE-only nDCG + latency |
| `ec2_latency.py` | FDE-only vs FDE+rerank server-side latency (avg/p50/p95/p99) |
| `ec2_breakdown.py` | Splits query latency into encode+prefetch vs rerank |
| `oversample_sweep.py` | Rescore-depth sweep on an existing index (warmup + `_source` excludes) |
| `ir_ksim.py` | IRPAPERS FDE-only / rerank recall across k_sim and rescore depth (ef_search=1024) |
| `ir_meanpool.py` | IRPAPERS mean-pool + rerank baseline (client-side), recall@k |
| `quant_multivector_test.py` | Client-side float32/fp16/int8 multivector quantization test |
| `ec2_fdeonly_ndcg.py` | FDE-only nDCG helper |

Scripts live in [`scripts/`](scripts/) (see [`scripts/README.md`](scripts/README.md) for
prerequisites and conventions); our raw run outputs are in [`results/`](results/) so you can
diff your reproduction against ours. All scripts talk to a local cluster on `127.0.0.1:9210` and
read data from `data_<dataset>/`. Adjust host/paths at the top of each script.

### Repo layout
```
benchmarks/
  BENCHMARK_RESULTS.md   ← this file
  scripts/               ← Python harnesses (+ README)
  results/               ← raw JSON outputs from our runs
```

---

## 10. Caveats (read before quoting numbers)

1. **Engine:** text quality from a faiss run, retest latency/levers from a lucene run — ~1–2 pt offset.
2. **Single-shard latency:** absolute ms are single-shard; rerank parallelizes across shards, so a
   multi-shard cluster would be faster. The *relative* lever effects are shard-invariant.
3. **Exact is a model ceiling,** not a perfect oracle; absolute values reflect dataset/metric difficulty.
4. **Quantization compact size is potential** (binary doc-values), not what `_source` realizes today.
5. **IRPAPERS metric is recall@10** (binary qrels), not nDCG — don't mix it under an nDCG header.
