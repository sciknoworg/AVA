# Knowledge Graph Link Prediction with Nomic Encoder

## Overview

This project performs **link prediction** on knowledge graphs using a transformer-based encoder.

Given a knowledge graph triple (head, relation, tail), the model ranks candidate entities to predict missing links.

---

## Task

Link Prediction

The model evaluates how well it can predict missing head or tail entities.

Metrics used:

- Hits@1
- Hits@3
- Hits@5
- Hits@10
- Mean Reciprocal Rank (MRR)

---

## Installation

pip install -r requirements.txt

---

## Example Usage

python -u run_nomic_link_prediction.py \
  --task_name kg \
  --data_dir data/FB15K \
  --output_dir output/fb15k_run1 \
  --do_train \
  --do_eval \
  --do_predict \
  --model_name nomic-ai/nomic-embed-text-v1.5 \
  --trust_remote_code \
  --num_train_epochs 3

---

## Output

- pytorch_model.bin → trained model
- eval_results.txt → validation results
- test_results.txt → test results
- link_prediction_metrics.txt → ranking metrics