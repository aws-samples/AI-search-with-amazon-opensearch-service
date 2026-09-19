# MUVERA + Late-Interaction Rerank on Amazon OpenSearch Service (No Plugin)

This tutorial shows how to run **MUVERA fixed-dimensional-encoding (FDE) retrieval with
late-interaction (ColBERT/ColPali) reranking on Amazon OpenSearch Service (AOS) without
installing any custom plugin.**

The approach ("Option X"): the **client** encodes both documents and queries into FDE
vectors using `fastembed`'s `Muvera`, indexes them into an ordinary `knn_vector` field,
and reranking is done by the **built-in `lateInteractionScore` Painless function** that
ships with the k-NN plugin (preinstalled on AOS). Nothing custom is installed on the
cluster.

> Verified on a live AOS domain running **OpenSearch 3.7** (`fastembed 0.7.4`,
> `colbert-ir/colbertv2.0`). The end-to-end flow — client FDE encode → `knn` prefetch →
> `rescore` → `lateInteractionScore` — returned correct topic rankings.

---

## Why no plugin is needed on AOS

| Piece | Where it runs | Provided by |
|---|---|---|
| Document FDE encoding | **Client** (`muvera.process_document`) | `fastembed` |
| Query FDE encoding | **Client** (`muvera.process_query`) | `fastembed` |
| ANN prefetch on FDE | Cluster | core `knn` query on a `knn_vector` field |
| Late-interaction rerank | Cluster | `lateInteractionScore` (k-NN plugin, **preinstalled on AOS**) |

Because the encoders run client-side, there is no `IngestPlugin` / `SearchPipelinePlugin`
to upload or get validated by AOS. The only cluster-side dependency is
`lateInteractionScore`, which is part of the k-NN plugin.

---

## Prerequisites

1. **AOS domain on OpenSearch 3.3 or later.** `lateInteractionScore` was introduced in 3.3.
   Confirm on your domain with the probe in the Appendix.
2. **k-NN enabled** (default on AOS standard distributions).
3. Client environment with:
   ```bash
   pip install fastembed opensearch-py numpy
   ```
4. Credentials for the domain (master user, or SigV4 / IAM — this tutorial uses basic auth
   for brevity; use IAM/SigV4 in production).

---

## Step 0 — Choose MUVERA parameters (and respect the dimension cap)

The FDE output dimension is:

```
FDE_DIM = r_reps * (2 ^ k_sim) * dim_proj
```

**A `knn_vector` field has a maximum `dimension` of 16,000.** Pick parameters that stay
under it.

| k_sim | dim_proj | r_reps | FDE_DIM | Under 16,000? |
|------:|---------:|-------:|--------:|:--:|
| 4 | 16 | 20 | **5,120** | ✅ (used in this tutorial) |
| 4 | 8  | 20 | 2,560 | ✅ |
| 6 | 32 | 20 | 40,960 | ❌ exceeds cap |

> The `fastembed` defaults (`k_sim=6, dim_proj=32, r_reps=20`) produce **40,960**, which
> exceeds the cap. Lower them for OpenSearch.

**Parity rule:** the **same** `k_sim`, `dim_proj`, `r_reps`, and `random_seed` must be used
for indexing and querying. Different values (or a different encoder implementation) produce
incompatible FDEs and silently destroy recall. Use one `Muvera` instance for both sides.

---

## Step 1 — Initialize the client-side encoder

```python
import numpy as np
from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera

# ColBERT late-interaction model (128-dim token vectors)
model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")

# Wrap with MUVERA. Keep these params identical for index + query.
muvera = Muvera.from_multivector_model(
    model=model,
    k_sim=4,
    dim_proj=16,
    r_reps=20,
    random_seed=42,
)

FDE_DIM = muvera.embedding_size   # 5120 with the params above
print("FDE dimension:", FDE_DIM)

def encode_document(text: str):
    mv = np.array(list(model.embed([text]))[0])   # [num_tokens][128] multi-vector
    fde = np.asarray(muvera.process_document(mv), dtype=float)  # normalize + fill
    return mv, fde

def encode_query(text: str):
    mv = np.array(list(model.embed([text]))[0])
    fde = np.asarray(muvera.process_query(mv), dtype=float)     # raw sum (NO normalize/fill)
    return mv, fde
```

> **Document vs query are different transforms.** Use `process_document` when indexing and
> `process_query` when searching. Do not use `process_document` for the query.

---

## Step 2 — Create the index

Two fields matter:

- `muvera_fde` — the FDE, a `knn_vector` (`dimension = FDE_DIM`, `space_type: innerproduct`).
- `colbert_vectors` — the **original** multi-vectors, stored as `object` with
  `enabled: false` so they stay in `_source` (where `lateInteractionScore` reads them) but
  are not indexed.

```
PUT /muvera_docs
{
  "settings": { "index.knn": true },
  "mappings": {
    "properties": {
      "text":            { "type": "text" },
      "colbert_vectors": { "type": "object", "enabled": false },
      "muvera_fde": {
        "type": "knn_vector",
        "dimension": 5120,
        "method": { "name": "hnsw", "engine": "lucene", "space_type": "innerproduct" }
      }
    }
  }
}
```

> `space_type` here (`innerproduct`) must match the `space_type` you pass to
> `lateInteractionScore` at query time. ColBERT MaxSim is an inner-product interaction.

---

## Step 3 — Index documents (client-encoded)

For each document, the client computes the FDE and indexes both the FDE and the raw
multi-vectors:

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "<YOUR_AOS_ENDPOINT_HOST>", "port": 443}],
    http_auth=("<USER>", "<PASSWORD>"),   # or use SigV4/IAM in production
    use_ssl=True, verify_certs=True,
)

docs = { "d1": "…", "d2": "…", "d3": "…" }   # id -> text

for doc_id, text in docs.items():
    mv, fde = encode_document(text)
    client.index(index="muvera_docs", id=doc_id, refresh=True, body={
        "text": text,
        "colbert_vectors": mv.tolist(),   # raw multi-vectors (for rerank)
        "muvera_fde": fde.tolist(),       # FDE (for ANN prefetch)
    })
```

The equivalent raw request per document:

```
PUT /muvera_docs/_doc/d1?refresh=true
{
  "text": "The central bank raised interest rates to control inflation.",
  "colbert_vectors": ${DOC_COLBERT_MULTIVECTORS},   // [[..128..], [..128..], ...]
  "muvera_fde":      ${DOC_FDE_VECTOR}              // [ .. 5120 floats .. ]
}
```

Placeholders:
- `${DOC_COLBERT_MULTIVECTORS}` = `model.embed([text])[0].tolist()` — list of 128-dim token vectors.
- `${DOC_FDE_VECTOR}` = `muvera.process_document(mv).tolist()` — 5120 floats.

---

## Step 4 — Search: `knn` prefetch + `rescore` late-interaction rerank

This is the recommended two-phase pattern (matches the OpenSearch late-interaction doc):
`knn` retrieves candidates on the FDE, then `rescore` recomputes scores with
`lateInteractionScore` over the top `window_size` only.

```python
q_mv, q_fde = encode_query("central bank monetary policy and inflation")

body = {
  "query": {
    "knn": { "muvera_fde": { "vector": q_fde.tolist(), "k": 100 } }
  },
  "rescore": {
    "window_size": 100,
    "query": {
      "rescore_query": {
        "script_score": {
          "query": { "match_all": {} },
          "script": {
            "source": "lateInteractionScore(params.query_vector, 'colbert_vectors', params._source, params.space_type)",
            "params": { "query_vector": q_mv.tolist(), "space_type": "innerproduct" }
          }
        }
      }
    }
  },
  "size": 10,
  "_source": { "excludes": ["muvera_fde", "colbert_vectors"] }
}
res = client.search(index="muvera_docs", body=body)
```

The equivalent raw request with placeholders:

```
POST /muvera_docs/_search
{
  "query": {
    "knn": {
      "muvera_fde": {
        "vector": ${QUERY_FDE_VECTOR},          // muvera.process_query(q_mv) -> 5120 floats
        "k": 100
      }
    }
  },
  "rescore": {
    "window_size": 100,
    "query": {
      "rescore_query": {
        "script_score": {
          "query": { "match_all": {} },
          "script": {
            "source": "lateInteractionScore(params.query_vector, 'colbert_vectors', params._source, params.space_type)",
            "params": {
              "query_vector": ${QUERY_COLBERT_MULTIVECTORS},   // [[..128..], ...] raw query tokens
              "space_type": "innerproduct"
            }
          }
        }
      }
    }
  },
  "size": 10,
  "_source": { "excludes": ["muvera_fde", "colbert_vectors"] }
}
```

Placeholders:
- `${QUERY_FDE_VECTOR}` = `muvera.process_query(q_mv).tolist()` — 5120 floats. Drives ANN prefetch.
- `${QUERY_COLBERT_MULTIVECTORS}` = `model.embed([query])[0].tolist()` — raw query token vectors. Drives the exact MaxSim rerank.

### How the two phases combine

- **`knn`** — fast approximate retrieval on the FDE; produces the candidate set and the
  first-pass score.
- **`rescore`** — recomputes `lateInteractionScore` (exact MaxSim over the raw
  `colbert_vectors`) on the top `window_size` candidates, and **blends** it with the knn
  score. Tune the balance with `query_weight` / `rescore_query_weight` inside `rescore` if
  desired (defaults are 1.0 each).

> **`script_score`-as-main-query alternative (not recommended for production):** wrapping
> the `knn` in a top-level `script_score` computes `lateInteractionScore` on *every* match
> and *replaces* the ANN score entirely. It works but is more expensive and discards the
> ANN signal. Prefer `rescore`.

---

## `lateInteractionScore` signature notes

Two overloads exist:

```
lateInteractionScore(query_vectors, doc_field_name, _source)              // 3-arg: defaults to L2
lateInteractionScore(query_vectors, doc_field_name, _source, space_type)  // 4-arg: explicit metric
```

**Use the 4-arg form with `"innerproduct"`** for ColBERT/ColPali. The 3-arg form defaults
to L2, which is the wrong metric for MaxSim and will degrade relevance.

- `query_vectors` — the **raw** query multi-vectors (`List<List<Number>>`), not the FDE.
- `doc_field_name` — `'colbert_vectors'`, read from `_source`.
- `_source` — pass `params._source`.
- `space_type` — one of `innerproduct`, `cosinesimil`, `l2`, `l1`, `linf`.

---

## Common pitfalls (all silent failures)

1. **FDE dimension > 16,000** — index creation fails or vectors are rejected. Keep
   `r_reps * 2^k_sim * dim_proj ≤ 16,000`.
2. **Encoder param / seed mismatch between index and query** — recall silently collapses.
   Use one `Muvera` instance for both, with a pinned `random_seed`.
3. **Mixing encoder implementations** — `fastembed`'s MUVERA and the k-NN Java
   `MuveraEncoder` are different implementations; their FDEs are **not** interchangeable.
   If you index with one, query with the same one.
4. **`colbert_vectors` not in `_source`** — if the field is indexed/removed instead of
   `object` + `enabled:false`, `lateInteractionScore` throws "Document vectors cannot be
   null or empty".
5. **`space_type` mismatch** between the mapping and the `lateInteractionScore` arg — scores
   computed under the wrong metric.
6. **Using `process_document` for the query** (or vice versa) — wrong FDE; the two
   transforms are asymmetric by design.

---

## Appendix — Verify `lateInteractionScore` exists on your domain

A minimal read/write probe (delete the probe index afterward):

```
PUT /li_probe
{ "mappings": { "properties": { "cv": { "type": "object", "enabled": false } } } }

PUT /li_probe/_doc/1?refresh=true
{ "cv": [[1,0],[0,1]] }

POST /li_probe/_search
{
  "query": {
    "script_score": {
      "query": { "match_all": {} },
      "script": {
        "source": "lateInteractionScore(params.q, 'cv', params._source, 'innerproduct')",
        "params": { "q": [[1,0],[0,1]] }
      }
    }
  }
}

DELETE /li_probe
```

If the search returns a score (rather than a "unknown function" error), the function is
available and your domain supports this tutorial.

---

## Security note

Use IAM/SigV4 auth for AOS in production rather than basic auth, and never commit domain
credentials. If credentials were used in a shared/interactive context, rotate them.
