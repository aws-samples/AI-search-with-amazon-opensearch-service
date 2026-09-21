# Benchmark scripts

Python harnesses that reproduce the numbers in [`../BENCHMARK_RESULTS.md`](../BENCHMARK_RESULTS.md).

## Prerequisites

- A local **OpenSearch 3.8.0** on `127.0.0.1:9210` with the **opensearch-muvera-plugin** and
  **k-NN** installed (see the repo root README to build/install the plugin).
- Python 3.9+ with:
  ```bash
  pip install opensearch-py numpy ijson fastembed
  ```
- Dataset embeddings under `data_<dataset>/` next to the scripts (or edit the `DATA`/`D` path
  at the top of each script):
  ```
  data_scifact/  data/ (nfcorpus)  data_irpapers/
    doc_embeddings.json  query_embeddings.json  qrels.json
  ```
  Generate these by encoding the corpus + queries with a late-interaction model (ColBERTv2 for
  text, ColModernVBERT for IRPAPERS) and dumping `{id: {embeddings: [[128]xN]}}`.

## Conventions

- All scripts connect to `127.0.0.1:9210` (edit the `OpenSearch(hosts=...)` line to change).
- Indices are created on first run and **reused** if present (delete to rebuild).
- FDE params must match between ingest and query: `dim, k_sim, dim_proj, r_reps, seed` — the
  scripts pass these as CLI args; keep them consistent.

## What each script does

| Script | Command | Output |
|---|---|---|
| `run_nfcorpus_retest.py` | `python run_nfcorpus_retest.py` | exact / mean-pool / FDE-only / MUVERA+rerank nDCG (text) |
| `ec2_variant_ds.py` | `python ec2_variant_ds.py <data_dir> <name> <k_sim> <dim_proj> <r_reps>` | one MUVERA variant: rerank & FDE-only nDCG + latency |
| `ec2_latency.py` | `python ec2_latency.py` | FDE-only vs FDE+rerank latency (avg/p50/p95/p99) |
| `ec2_breakdown.py` | `python ec2_breakdown.py` | encode+prefetch vs rerank split |
| `oversample_sweep.py` | `python oversample_sweep.py <data_dir> <index> <k_sim> <dim_proj> <r_reps> <oversamples csv>` | rescore-depth sweep (warmup + source-excludes) |
| `ir_ksim.py` | `python ir_ksim.py <name> <k_sim> <dim_proj> <r_reps> <fdeonly\|rerank> [rescore_ks_csv]` | IRPAPERS recall, ef_search=1024 |
| `ir_meanpool.py` | `python ir_meanpool.py` | IRPAPERS mean-pool + rerank recall (client-side) |
| `quant_multivector_test.py` | `python quant_multivector_test.py <data_dir> <max_docs\|0>` | float32/fp16/int8 multivector quantization test |
| `ec2_fdeonly_ndcg.py` | `python ec2_fdeonly_ndcg.py` | FDE-only nDCG helper |

Raw outputs for our runs are in [`../results/`](../results/) — compare against these to verify
your reproduction.
