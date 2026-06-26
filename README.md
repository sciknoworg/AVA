<div align="center">
  <img src="images/logo.png" alt="AVA Logo" width="220"/>
</div>

<div align="center">
  <h1>Do General NLP Embeddings Capture Ontological Reasoning?</h1>
</div>

<p align="center">
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="#"><img alt="Triplets" src="https://img.shields.io/badge/triplets-171%2C007-orange.svg"></a>
  <a href="#"><img alt="Ontologies" src="https://img.shields.io/badge/ontologies-163-orange.svg"></a>
</p>

---

## Overview

General-purpose sentence embeddings (E5, GTE, BGE, MPNet, Qwen3‑Embedding, OpenAI `text-embedding-3-*`, …) perform very well on Semantic Textual Similarity and retrieval benchmarks. **AVA** asks a sharper question: do these embeddings actually understand the *logical* structure of an ontology, or are they just matching surface lexical overlap?

AVA is a data‑generation pipeline that turns 163 heterogeneous OWL/RDFS ontologies into **171,007 contrastive triplets**. Each triplet consists of:

- **Anchor** — a sentence stating a fact directly asserted by the ontology (a subclass relation, a domain/range constraint, an equivalence, a disjointness, …).
- **Positive** — a semantically equivalent paraphrase of the anchor (different wording, same logical content).
- **Hard Negative** — a sentence that is *lexically very close* to the anchor but *logically wrong* — e.g. a sibling class swapped in for a subclass, an inverted property domain, or a disjoint concept substituted for the correct one.

Because the hard negative is constructed to be more lexically similar to the anchor than the positive is, a model cannot solve AVA by relying on surface overlap — it has to encode relational/ontological semantics.

```
Anchor:        An online gaming account is a subclass of an online account.
Positive:      The class of online gaming accounts is defined as a subclass
               within the online account hierarchy.
Hard Negative: An online gaming account is a subclass of an agent.
```

### Key findings

| | |
|---|---|
| 🏆 **Best pre-trained model** | Qwen3‑Embedding‑0.6B — 0.739 triplet accuracy, but only **0.572** hard‑negative accuracy |
| 📉 **General trend** | Most general-purpose embeddings sit at 0.39–0.66 triplet accuracy and collapse further (often <0.3) on lexically-deceptive hard negatives |
| 📈 **Fine-tuning works *too* well** | Contrastive triplet / hyperbolic fine-tuning pushes MPNet-base from 0.636 → **0.989** triplet accuracy |
| ⚠️ **But it doesn't transfer** | The same fine-tuned models show flat or *negative* gains on downstream taxonomy discovery and ontology alignment tasks |
| 🧭 **Takeaway** | High AVA accuracy reflects learned *perturbation-specific pattern recognition*, not robust, transferable ontology-level reasoning — contrastive discrimination and ontology generalization are not interchangeable measures of understanding |

---

## Repository structure

```
ava/
├── data/                          # Generated triplet dataset
│   ├── train.json                 # 153,211 triplets (70,703 hard negatives)
│   └── test.json                  # 17,796 triplets (7,229 hard negatives)
│
├── src/
│   ├── dataset/
│   │   ├── subgraphs.py           # Ontology → NetworkX graph → BFS/community/module subgraph extraction
│   │   └── synthesis.py           # Subgraph → LLM prompt → (anchor, positive, hard negative) synthesis
│   ├── evaluation.py              # evaluation metrics
│   ├── hyperbolic.py              # Poincaré-ball hyperbolic triplet loss
│   ├── dpo.py                     # Embedding-adapted Direct Preference Optimization loss
│   └── utils.py                   # Seeding, dataset building, weighted multi-loss combination, IO helpers
├── subgraph_extractor.py          # Step 1: extract BFS subgraphs from ontologies
├── synthesis_extractor.py         # Step 2: synthesize contrastive triplets from subgraphs via LLM
├── finetune_triplet.py            # Step 3: contrastive fine-tuning (Euclidean / hyperbolic / DPO)
├── evaluate.py                    # Step 4: benchmark pre-trained & fine-tuned embedding models
├── make_latex_table.py            # Render results/*.json into a LaTeX results table
├── run_experiments.sh             # End-to-end sweep: margins, hyperbolic curvatures, DPO betas
├── results/                       # Per-model evaluation outputs (JSON) + compiled LaTeX table
├── analysis/                      # Dataset analysis notebook, lexical/semantic similarity dumps, figures
├── experiments/                   # Taxonomy learning & ontology alignment downstream evaluation pipelines.
├── images/
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone <this-repo-url>
cd ava
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the core fine-tuning/evaluation pipeline (`sentence-transformers`, `torch`, `transformers`, `faiss-cpu`, `numpy`, `pandas`, `tqdm`, `nltk`, `spacy`). Depending on which stage you run, you will also need:

| Stage | Extra dependencies |
|---|---|
| Subgraph extraction (`subgraph_extractor.py`) | `rdflib`, `networkx`, `python-louvain` (`community`), [`ontolearner`](https://github.com/) for ontology loading |
| Triplet synthesis (`synthesis_extractor.py`) | `openai`, `python-dotenv`, `certifi`, plus a local/HF-hosted LLM for generation (default config targets `Qwen/Qwen3.5-35B-A3B`) |
| Evaluation against the OpenAI embedding API (`evaluate.py`) | `openai`, an `OPENAI_API_KEY`/`OPENAI_KEY` environment variable |
| Downstream baselines (`experiments/`) | see the individual READMEs under `experiments/triple-classification` and `experiments/link-prediction` (e.g. `pytorch_pretrained_bert` for KG-BERT/exBERT) |

> A `data/` and `results/` folder with `.gitkeep` placeholders are already provided so generated artifacts have a home.

---

## The AVA dataset

### Construction pipeline

1. **Ontology graph extraction** — 163 ontologies spanning biomedical (GO, OBI), geospatial, social, engineering, and schema domains are parsed into undirected graphs where nodes are OWL classes/properties and edges are RDFS/OWL relations (`rdfs:subClassOf`, `owl:domain`, `owl:range`, reified existential restrictions, …).
2. **Subgraph sampling** — two-hop breadth-first search seeded at every class yields subgraphs of 3–20 nodes, deduplicated by MD5 fingerprint of their node sets → **50,548** unique subgraphs.
3. **LLM-based triplet synthesis** — each subgraph is rendered as a structured prompt (entities + definitions + RDF triples) and passed to an LLM, which is instructed to emit five (anchor, positive, hard negative) triplets per subgraph, covering hierarchy assertions, domain/range constraints, equivalence, and disjointness.
4. **Quality filtering** (token-set ratio via RapidFuzz):
   - Anchor–negative similarity ≥ 90 → flagged as a *hard negative* and routed to a dedicated evaluation slice.
   - Positive–negative similarity ≥ 90 → discarded (too close to be meaningfully distinguished).
5. **Ontology-level train/test split** — all subgraphs from a given ontology stay in either train or test (never both), and the test set is restricted to ontologies sharing fewer than 100 labels with the training set, enforcing a genuine **cross-ontology generalization** setting.

### Statistics

| | Count |
|---|---|
| Ontologies | 163 |
| BFS subgraphs | 50,548 |
| Synthesized samples (raw) | 197,326 |
| Removed (positive–negative too similar) | 4,154 |
| **After cleaning & deduplication** | **171,007** |
| Hard negatives (anchor–negative similarity ≥ 90) | 77,932 |
| Mean sentence length | 10 words |
| Train split (hard negatives) | 153,211 (70,703) |
| Test split (hard negatives) | 17,796 (7,229) |

### Format

`data/train.json` and `data/test.json` are JSON lists of records:

```json
{
  "anchor": "An online gaming account is a subclass of an online account.",
  "positive": "The class of online gaming accounts is defined as a subclass within the online account hierarchy.",
  "negative": "An online gaming account is a subclass of an agent.",
  "subgraph_id": "bfs_bb6ca90f",
  "source": "FOAF",
  "anchor_positive_score": 83.52,
  "anchor_negaive_score": 94.23,
  "positive_negaive_score": 85.39
}
```

`*_score` fields are RapidFuzz token-set-ratio lexical similarities (0–100) used for hard-negative flagging and quality filtering.

---

## Usage

### 1. Generate your own triplets from new ontologies (optional)

The released `data/train.json` / `data/test.json` already contain the full 171k-triplet dataset, so this step is only needed if you want to extend AVA to additional ontologies.

```bash
# Step 1 — extract BFS subgraphs for every ontology available via `ontolearner`
python subgraph_extractor.py

# Step 2 — synthesize (anchor, positive, hard negative) triplets with an LLM
#   requires OPENAI_KEY in your environment (or a .env file) for JSON post-processing,
#   plus a local/HF generator model (default: Qwen/Qwen3.5-35B-A3B)
python synthesis_extractor.py
```

`synthesis_extractor.py` is designed to be run as multiple parallel workers — it claims one un-processed ontology at a time via `worker_*.json` lock files, so several processes can be launched concurrently without duplicating work.

### 2. Fine-tune an embedding model

```bash
python finetune_triplet.py \
    --train_path data/train.json \
    --model_id   sentence-transformers/all-mpnet-base-v2 \
    --output_dir assets/mpnet-triplet \
    --num_train_epochs 10 \
    --per_device_train_batch_size 256 \
    --margin 0.3 \
    --fp16
```

Add `--hyperbolic --hyperbolic_c 0.3 --w_hyperbolic 0.7` for the Poincaré-ball objective, or `--dpo --rl_beta 0.5 --w_dpo 0.3` for the embedding-adapted DPO objective. The two can be combined (as in the paper's best configuration). Key flags:

| Flag | Description | Default |
|---|---|---|
| `--margin` | Triplet-loss margin | `0.3` |
| `--hyperbolic` / `--hyperbolic_c` / `--w_hyperbolic` | Enable hyperbolic loss / Poincaré curvature / loss weight | off / `0.1` / `0.1` |
| `--dpo` / `--rl_beta` / `--w_dpo` | Enable DPO-style loss / temperature β / loss weight | off / `0.1` / `0.1` |
| `--fp16` | Mixed-precision training | off |

`run_experiments.sh` reproduces the full paper sweep: triplet-loss margins {0.3, 0.5, 0.7}, hyperbolic curvatures {0.1, 0.3, 0.5, 1.0}, DPO betas {0.1, 0.3, 0.5}, and the combined hyperbolic+DPO run, followed by evaluation.

### 3. Evaluate embedding models on AVA

```bash
python evaluate.py
```

This benchmarks every model listed in `models_to_evaluate` inside `evaluate.py` — 25+ pre-trained checkpoints (E5, GTE, BGE, MiniLM, MPNet, Qwen3‑Embedding, OpenAI `text-embedding-3-*`, Nomic, Llama-Embed-Nemotron, Linq-Embed-Mistral, …) plus any locally fine-tuned checkpoints under `assets/`, and writes one JSON report per model to `results/`. Edit the `models_to_evaluate` list to add/remove models.

Each report contains, per split:

```json
{
  "triplet_accuracy": 0.739,
  "MRR": 0.84,
  "Recall@1": 0.739,
  "hard_negative_acc": 0.572,
  "accuracy": 0.71, "precision": 0.69, "recall": 0.75, "f1": 0.72,
  "roc_auc": 0.77, "avg_precision": 0.74,
  "best_threshold": 0.82
}
```

- **Triplet accuracy** — fraction of triplets where `sim(anchor, positive) > sim(anchor, negative)` (ties count as incorrect).
- **Hard-negative accuracy** — the same metric restricted to the subset where the negative is *lexically* closer to the anchor than 90 (token-set ratio) — the regime where surface-similarity shortcuts fail.
- **MRR / Recall@k** — retrieval-style ranking metrics.
- **Classification metrics** (`accuracy`, `precision/recall/F1`, `ROC-AUC`, `avg_precision`) — treating anchor–positive vs. anchor–negative as a binary same/different-meaning classification problem at a tuned cosine-similarity threshold.

### 4. Downstream transfer evaluation

The `experiments/` directory hosts the two downstream Semantic Web tasks used to test whether AVA-tuned embeddings generalize beyond the benchmark itself:

- **Ontology Alignment** - aligning classes across ontologies (ENVO–SWEET, Mouse–Human, MaterialInformation–MatOnto, Yago–Wikidata).
- **Taxonomy Learning** - discovering subclass relations within a single ontology (GO, SchemaOrg, SWEET, OBI, PO).


## License

Released under the [MIT License](LICENSE).