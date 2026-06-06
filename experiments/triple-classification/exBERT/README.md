# exBERT

A transformer-based model for performing triple classification on knowledge graphs.

---

## How to use exBERT

You can set up the code locally and run it using the commands below.

---

## Requirements

Install the required Python packages:

```bash
pip install scikit-learn torch transformers accelerate>=1.1.0
```

### GPU Setup (Recommended)

To enable GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Make sure your system has a compatible NVIDIA GPU and drivers installed.

---

## Dataset Setup

Make sure the dataset folders (e.g., `PWC21`, `UMLS`) are placed in the same directory as the code, under a `data/` folder.

---

## Running exBERT

To run the **exBERT** script, use the following command.  
Changing the parameters allows you to select dataset, task, and hyperparameters.

---

### PWC21 (SciBERT, Triple Classification)

```bash
python exBERT.py --task tc `
--do_train `
--do_eval `
--do_predict `
--data_dir .\data\PWC21 `
--bert_model allenai/scibert_scivocab_uncased `
--max_seq_length 64 `
--train_batch_size 8 `
--learning_rate 5e-5 `
--num_train_epochs 3.0 `
--output_dir .\output_PWC21_TC_test\ `
--gradient_accumulation_steps 1 `
--eval_batch_size 32
```

---

### UMLS (SciBERT, Triple Classification)

```bash
python exBERT.py --task tc `
--do_train `
--do_eval `
--do_predict `
--data_dir .\data\UMLS `
--bert_model allenai/scibert_scivocab_uncased `
--max_seq_length 32 `
--train_batch_size 8 `
--learning_rate 5e-5 `
--num_train_epochs 3.0 `
--output_dir .\output_UMLS_TC_test\ `
--gradient_accumulation_steps 1 `
--eval_batch_size 32
```
