# Nomic Triple Classifier

A transformer-based embedding model for triple classification using Nomic embeddings.

---

## How to use

You can set up the code locally and run it using the commands below.

---

## Requirements

Install the required Python packages:

```bash
pip install torch tqdm transformers numpy einops
```

### GPU Setup (Recommended)

To enable GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Make sure your system has a compatible NVIDIA GPU and drivers installed.

---

## Dataset Setup

Make sure the dataset folders (e.g., `PWC21`, `UMLS`) are placed in the same directory as the code.

---

## Running the Model

### PWC21 (Triple Classification)

```bash
python run_nomic_triple_classifier.py `
  --data_dir .\PWC21 `
  --bert_model nomic-ai/nomic-embed-text-v1.5 `
  --task_name kg `
  --output_dir .\output_nomic_PWC21 `
  --do_train `
  --do_eval `
  --do_predict `
  --max_seq_length 64 `
  --train_batch_size 8 `
  --eval_batch_size 32 `
  --learning_rate 5e-5 `
  --num_train_epochs 3
```

---

### UMLS (Triple Classification)

```bash
python run_nomic_triple_classifier.py `
  --data_dir .\UMLS `
  --bert_model nomic-ai/nomic-embed-text-v1.5 `
  --task_name kg `
  --output_dir .\output_nomic_umls `
  --do_train `
  --do_eval `
  --do_predict `
  --max_seq_length 32 `
  --train_batch_size 8 `
  --eval_batch_size 32 `
  --learning_rate 5e-5 `
  --num_train_epochs 3
```
