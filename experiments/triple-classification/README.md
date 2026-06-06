# Triple Classification Experiments

This directory contains implementations for triple classification using three models:

- exBERT
- KG-BERT
- Nomic embedding-based classifier

---

## Task Overview

Triple classification is a task where the model predicts whether a given knowledge graph triple (head, relation, tail) is valid or corrupted.

---

## Datasets

We use two standard datasets:

- **PWC21** – derived from Papers With Code  
- **UMLS** – biomedical knowledge graph  

Datasets are not included in this repository.  
They can be downloaded from the links provided in `data/README.md`.

---

## Models

Each model is implemented in its own directory:

- `exbert/`
- `kgbert/`
- `nomic/`

Each folder contains:
- code files  
- instructions to run the model  

---

## Experimental Settings

| Parameter            | PWC21 | UMLS |
|---------------------|------|------|
| Max Sequence Length | 64   | 32   |
| Train Batch Size    | 8    | 8    |
| Eval Batch Size     | 32   | 32   |
| Learning Rate       | 5e-5 | 5e-5 |
| Epochs              | 3    | 3    |

---

## Evaluation Metrics

Evaluation is based on ranking metrics:

- **MRR (Mean Reciprocal Rank)** – average inverse rank of the correct triple  
- **Hits@K (1, 3, 5, 10)** – whether the correct triple appears in top-K predictions  

Each test triple is evaluated against **41 candidates**:
- 1 positive triple  
- 20 head corruptions  
- 20 tail corruptions  

---

## Results

| Model   | **UMLS** |  |  |  |  | **PWC** |  |  |  |  |
|---------|----------|------|------|------|--------|----------|------|------|------|--------|
|         | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
| exBERT  | 0.203 | 0.023 | 0.210 | 0.410 | 0.740 | 0.423 | 0.232 | 0.501 | 0.683 | 0.895 |
| KG-BERT | 0.329 | 0.110 | 0.407 | 0.646 | 0.891 | 0.157 | 0.063 | 0.139 | 0.201 | 0.362 |
| Nomic   | 0.331 | 0.091 | 0.439 | 0.699 | 0.918 | 0.151 | 0.071 | 0.135 | 0.162 | 0.290 |
