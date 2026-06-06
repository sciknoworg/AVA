# KG-BERT

A transformer-based model for triple classification using BERT representations of knowledge graph triples.

---

## How to use

You can set up the code locally and run it using the commands below.

---

## Requirements

Install the required Python packages:

```bash
pip install torch numpy scikit-learn tqdm boto3 requests pytorch-pretrained-bert>=0.6.2 Wikipedia
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
python run_bert_triple_classifier.py `
  --data_dir .\PWC21 `
  --bert_model bert-base-uncased `
  --task_name kg `
  --output_dir .\output_PWC21_TC_test\ `
  --do_train `
  --do_eval `
  --do_predict `
  --do_lower_case `
  --max_seq_length 64 `
  --train_batch_size 8 `
  --eval_batch_size 32 `
  --learning_rate 5e-5 `
  --num_train_epochs 3
```

---

### UMLS (Triple Classification)

```bash
python run_bert_triple_classifier.py `
  --data_dir .\UMLS `
  --bert_model bert-base-uncased `
  --task_name kg `
  --output_dir .\output_UMLS_TC_test\ `
  --do_train `
  --do_eval `
  --do_predict `
  --do_lower_case `
  --max_seq_length 32 `
  --train_batch_size 8 `
  --eval_batch_size 32 `
  --learning_rate 5e-5 `
  --num_train_epochs 3
```
