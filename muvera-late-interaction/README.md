# MUVERA late-interaction retrieval for OpenSearch

Scale **late-interaction** retrieval (ColBERT / ColPali multi-vectors) on self-managed
OpenSearch using **MUVERA** (Multi-Vector Retrieval via Fixed-Dimensional Encodings) as a
two-phase **retrieve → rerank** pattern:

- **Phase 1 (retrieve):** each document's multi-vectors are encoded into a single
  **Fixed Dimensional Encoding (FDE)** whose dot product approximates MaxSim, and indexed as a
  `knn_vector` — so phase 1 is ordinary, fast ANN search.
- **Phase 2 (rerank):** the top candidates are rescored with **true MaxSim** over their stored
  multi-vectors, using k-NN's `lateInteractionScore` scripting function.

This gives near-exact late-interaction quality while keeping retrieval at ANN speed. The
`muvera` (ingest) and `muvera_query` (search) processors are provided by the standalone
**`opensearch-muvera-plugin`** in this folder.

> Companion notebook in this repo: **`ColPali with OpenSearch.ipynb`** (late-interaction / ColPali).
> For **Amazon OpenSearch Service (managed)**, where custom ingest-processor plugins aren't
> installable, use the client-side FDE path in [`docs/MUVERA-late-interaction-AOS-tutorial.md`](docs/MUVERA-late-interaction-AOS-tutorial.md).

---

## Get the code

Clone this branch and move into the plugin folder:

```bash
git clone --branch muvera-late-interaction --single-branch \
  https://github.com/aws-samples/AI-search-with-amazon-opensearch-service.git

cd AI-search-with-amazon-opensearch-service/muvera-late-interaction
```

Everything referenced below is relative to this `muvera-late-interaction/` folder.

---

## What's in this folder

```
muvera-late-interaction/
├── README.md                 ← you are here
├── plugin/                   ← plugin source (build it for your OpenSearch version)
│   ├── src/main/java/org/opensearch/muvera/*.java
│   ├── build.gradle, settings.gradle, gradlew, gradle/
│   └── LICENSE.txt, NOTICE.txt
├── dist/
│   └── opensearch-muvera-plugin-3.8.0.0.zip   ← prebuilt, for OpenSearch 3.8.0 ONLY
└── docs/
    └── MUVERA-late-interaction-AOS-tutorial.md
```

**Which do I use — the prebuilt zip or the source?**

- On **OpenSearch 3.8.0** → use the prebuilt `dist/…-3.8.0.0.zip` and skip straight to *Install*.
- On **any other version** → build from `plugin/` (below). OpenSearch enforces an **exact**
  `opensearch.version` match, so a 3.8.0 zip will not install on 3.7.x / 3.9.x.

---

## Requirements

- Self-managed **OpenSearch** cluster (OSS distribution), JDK 21.
- The **k-NN plugin** on every node (ships with OpenSearch). It provides the `knn_vector` field
  type and the `lateInteractionScore` rerank function.
- A late-interaction model to produce token vectors (e.g. ColBERTv2 128-dim, ColPali/ColModernVBERT).

---

## Build from source (for your OpenSearch version)

```bash
cd plugin
# edit build.gradle: set opensearch.version (and the plugin version) to your target, e.g. 3.9.0
./gradlew clean assemble
# artifact:
#   plugin/build/distributions/opensearch-muvera-plugin-<version>.zip
```

The provided `dist/opensearch-muvera-plugin-3.8.0.0.zip` was produced this same way against
OpenSearch 3.8.0.

---

## Install (upload) on the cluster

Install on **every node**, then restart (rolling restart for multi-node).

```bash
# from $OPENSEARCH_HOME on each node — local file:
bin/opensearch-plugin install --batch file:///path/to/opensearch-muvera-plugin-3.8.0.0.zip

# …or from a URL (S3 pre-signed / artifact server / GitHub raw):
bin/opensearch-plugin install --batch https://<host>/opensearch-muvera-plugin-3.8.0.0.zip
```

Docker:

```dockerfile
FROM opensearchproject/opensearch:3.8.0
COPY dist/opensearch-muvera-plugin-3.8.0.0.zip /tmp/
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch \
    file:///tmp/opensearch-muvera-plugin-3.8.0.0.zip
```

Verify:

```bash
bin/opensearch-plugin list          # → opensearch-muvera-plugin
curl -s "localhost:9200/_cat/plugins?v"   # opensearch-muvera-plugin + k-NN on every node
```

Uninstall / upgrade: `bin/opensearch-plugin remove opensearch-muvera-plugin` then restart;
to upgrade, remove → install new zip → restart (no in-place upgrade).

---

## Usage

FDE dimension = `r_reps × 2^k_sim × dim_proj` (cap 16,000). The `dim`, `k_sim`, `dim_proj`,
`r_reps`, and `seed` **must match** between the ingest and query processors so document and
query FDEs are compatible.

**1) Ingest pipeline + mapping**

```json
PUT _ingest/pipeline/muvera-ingest
{ "processors": [{ "muvera": {
  "dim": 128, "k_sim": 5, "dim_proj": 16, "r_reps": 20, "seed": 42,
  "source_field": "colbert_vectors", "target_field": "muvera_fde"
}}]}

PUT muvera-index
{ "settings": { "index.knn": true, "default_pipeline": "muvera-ingest" },
  "mappings": { "properties": {
    "colbert_vectors": { "type": "object", "enabled": false },
    "muvera_fde": { "type": "knn_vector", "dimension": 10240,
      "method": { "name": "hnsw", "engine": "lucene", "space_type": "innerproduct" } }
}}}
```

**2) Search pipeline** (encodes the query multi-vectors to an FDE)

```json
PUT _search/pipeline/muvera-search
{ "request_processors": [{ "muvera_query": {
  "dim": 128, "k_sim": 5, "dim_proj": 16, "r_reps": 20, "seed": 42,
  "target_field": "muvera_fde"
}}]}
```

**3) Query — FDE retrieve + MaxSim rerank**

```json
POST muvera-index/_search?search_pipeline=muvera-search
{ "query": { "template": { "script_score": {
    "query": { "knn": { "muvera_fde": { "vector": "${muvera_fde}", "k": 40 } } },
    "script": { "source":
      "lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)",
      "params": { "query_vectors": [[/* query token vectors */]], "space_type": "innerproduct" } }
}}},
  "size": 10, "_source": { "excludes": ["colbert_vectors","muvera_fde"] } }
```

`k` (= `size × oversample`) is the phase-1 shortlist / **rescore depth** — the main latency knob.

---

## Tuning notes (from benchmarking)

- **Rescore depth is the primary latency lever** — it scales ~linearly. On well-fitting text
  corpora you can halve it (e.g. depth 40 → 20) for ~40% lower latency at a small quality cost.
- **A higher-quality FDE (more `r_reps`) lets you rerank shallower at the same quality** —
  low sampling and FDE tuning are complementary.
- **Quantizing the multi-vectors (int8/fp16)** is a *memory* lever (up to ~4× smaller) at ~0
  quality cost, but does **not** reduce rerank latency in the `_source`-based scoring path.
- Dense visual docs (~1,000 vectors/page) need **deeper** rescore; FDE-only retrieval is not
  sufficient for them.

---

---

## Amazon OpenSearch Service (managed): client-side FDE with FastEmbed

Amazon OpenSearch Service does **not** allow installing this custom ingest-processor plugin, so
the `muvera` / `muvera_query` processors are unavailable. Instead, compute the FDE **on the
client** with [FastEmbed](https://github.com/qdrant/fastembed) (0.7.2+) and index it as a plain
`knn_vector`. Everything else — the two-phase pattern, the `knn` retrieve, and the
`lateInteractionScore` rerank — works unchanged, because the k-NN plugin ships with AOS.

**What changes vs. the self-managed plugin path**

| | Self-managed (plugin) | Amazon OpenSearch Service (client-side) |
|---|---|---|
| FDE for documents | `muvera` ingest processor (server-side) | FastEmbed `Muvera.process_document()` (client) |
| FDE for queries | `muvera_query` search processor (server-side) | FastEmbed `Muvera.process_query()` (client) |
| Index the FDE | `knn_vector` | `knn_vector` (same) |
| Rerank | `lateInteractionScore` | `lateInteractionScore` (same) |

> **Consistency rule:** documents and queries must be encoded with the **same** MUVERA
> parameters (`k_sim`, `dim_proj`, `r_reps`) and the same FastEmbed model, so their FDEs are
> comparable. FDE dimension = `r_reps × 2^k_sim × dim_proj`.

### 1) Install FastEmbed

```bash
pip install --upgrade "fastembed>=0.7.2"
```

### 2) Encode documents client-side (index time)

```python
import numpy as np
from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera

# Create a multi-vector model (ColBERT here) and wrap it with MUVERA
model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
muvera = Muvera.from_multivector_model(
    model=model,
    k_sim=6,
    dim_proj=32,
    r_reps=20,
)  # FDE dim = 20 * 2^6 * 32 = 40960  (keep <= 16000 for a knn_vector; lower the params if needed)

# Multi-vectors + FDE for a document
doc_text = "sample document text"
doc_multivectors = np.array(list(model.embed([doc_text])))[0]   # shape [num_tokens, dim]
doc_fde = muvera.process_document(doc_multivectors)             # shape [FDE dim]
```

> Note: the example params above give a 40,960-dim FDE, which exceeds the 16,000 `knn_vector`
> cap. For a directly-indexable FDE use e.g. `k_sim=4, dim_proj=16, r_reps=20` → 5,120, or
> `k_sim=5, dim_proj=16, r_reps=20` → 10,240.

### 3) Create the index (plain `knn_vector`, no ingest pipeline)

```json
PUT muvera-index
{ "settings": { "index.knn": true },
  "mappings": { "properties": {
    "colbert_vectors": { "type": "object", "enabled": false },
    "muvera_fde": { "type": "knn_vector", "dimension": 10240,
      "method": { "name": "hnsw", "engine": "lucene", "space_type": "innerproduct" } }
}}}
```

Index each document with **both** the client-computed FDE and its raw multi-vectors:

```json
PUT muvera-index/_doc/1
{ "muvera_fde": [/* doc_fde ... */],
  "colbert_vectors": [[/* token 1 ... */], [/* token 2 ... */]] }
```

### 4) Encode the query client-side and search (retrieve + rerank)

```python
query_text = "sample query"
q_multivectors = np.array(list(model.query_embed([query_text])))[0]
q_fde = muvera.process_query(q_multivectors)   # use process_query (sum, no empty-cluster fill)
```

```json
POST muvera-index/_search
{ "query": { "script_score": {
    "query": { "knn": { "muvera_fde": { "vector": [/* q_fde ... */], "k": 40 } } },
    "script": { "source":
      "lateInteractionScore(params.query_vectors,'colbert_vectors',params._source,params.space_type)",
      "params": { "query_vectors": [[/* query token vectors */]], "space_type": "innerproduct" } }
}},
  "size": 10, "_source": { "excludes": ["colbert_vectors","muvera_fde"] } }
```

Use `process_document()` for documents and `process_query()` for queries — they aggregate
clusters differently (documents average and fill empty clusters; queries sum and leave them
empty), matching the MUVERA paper. A worked, end-to-end example is in
[`docs/MUVERA-late-interaction-AOS-tutorial.md`](docs/MUVERA-late-interaction-AOS-tutorial.md).

---

## Credits & license

MUVERA: Dhulipala et al., Google Research (2024). FDE parameter intuition and the FastEmbed
snippet: [qdrant.tech/articles/muvera-embeddings](https://qdrant.tech/articles/muvera-embeddings).
Plugin licensed Apache-2.0 (see `plugin/LICENSE.txt`, `plugin/NOTICE.txt`).
