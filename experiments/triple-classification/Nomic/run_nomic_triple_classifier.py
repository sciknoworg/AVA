from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """A single training/test example for triple classification.

            Represents a knowledge graph triple in text form:
            (head, relation, tail) → label

            Attributes:            
            guid : str
                Unique identifier for the example
            text_a : str
                Head entity text
            text_b : str
                Relation text
            text_c : str
                Tail entity text
            label : str
                "1" (positive triple) or "0" (negative/corrupted)
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

        

class InputFeatures(object):
    """
    Processed features ready for model input.

    Attributes:

    input_ids : torch.Tensor
        Tokenized input ids
    attention_mask : torch.Tensor
        Attention mask for padding
    label_id : int
        Numeric label (0 or 1)
    """
    def __init__(self, input_ids, attention_mask, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


class DataProcessor(object):
    """
    Base class for dataset processing.

    Defines interface for loading train/dev/test data.
    """
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Read TSV file into list of rows.
        """
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                if line:
                    lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """
        Processor for Knowledge Graph triple classification.

        - Load triples from TSV
        - Convert entity/relation IDs → text
        - Generate negative samples (corruption)`
    """
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_labels(self, data_dir):
        """Binary classification: valid (1) or invalid (0) triple"""
        return ["0", "1"]

    @staticmethod
    def _load_ent2text(data_dir):
        """
        Load mapping: entity_id → entity_text
        """
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf8') as f:
            for line in f:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]
        return ent2text

    @staticmethod
    def _load_rel2text(data_dir):
        """
        Load mapping: relation_id → relation_text
        """
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r', encoding='utf8') as f:
            for line in f:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    rel2text[temp[0]] = temp[1]
        return rel2text

    @staticmethod
    def corrupt_head_tail(ent2text, entities, examples, i, line, lines_str_set, set_type, text_a, text_b, text_c):
        """
        Generate ONE negative sample by corrupting either head or tail.

        Strategy:
        - 50% chance: replace head
        - 50% chance: replace tail
        - Ensure new triple is not in original dataset
        """
        rnd = random.random()
        guid = "%s-%s" % (set_type + "_corrupt", i)
        if rnd <= 0.5:
            while True:
                tmp_ent_list = list(set(entities) - {line[0]})
                tmp_head = random.choice(tmp_ent_list)
                tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            tmp_head_text = ent2text[tmp_head]
            examples.append(InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c=text_c, label="0"))
        else:
            while True:
                tmp_ent_list = list(set(entities) - {line[2]})
                tmp_tail = random.choice(tmp_ent_list)
                tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                if tmp_triple_str not in lines_str_set:
                    break
            tmp_tail_text = ent2text[tmp_tail]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=tmp_tail_text, label="0"))

    @staticmethod
    def corrupt_all_head_tail(ent2text, entities, examples, i, line, set_type, text_a, text_b, text_c, corrupt_items=20):
        """
        Generate MANY negative samples for ranking evaluation.

        Used for test:
        - 20 corrupted heads
        - 20 corrupted tails

        Total group size = 1 positive + 40 negatives = 41
        """
        randomizer = random.Random(12)
        # Corrupt heads
        head_entities = list(set(entities) - {line[0]})
        for j, corrupt_head in enumerate(randomizer.sample(head_entities, corrupt_items)):
            guid = "%s-%s-head-%s" % (set_type + "_corrupt", i, j)
            tmp_head_text = ent2text[corrupt_head]
            examples.append(InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c=text_c, label="0"))
        # Corrupt tails
        tail_entities = list(set(entities) - {line[2]})
        for j, corrupt_tail in enumerate(randomizer.sample(tail_entities, corrupt_items)):
            guid = "%s-%s-tail-%s" % (set_type + "_corrupt", i, j)
            tmp_tail_text = ent2text[corrupt_tail]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=tmp_tail_text, label="0"))

    def _create_examples(self, lines, set_type, data_dir):
        """
            Convert raw TSV rows into `InputExample` objects.

            Processing Flow:

            1) Load entity and relation text mappings.
            2) Skip malformed or unmapped triples.
            3) Convert each triple id form into readable text form.
            4) Build positive and negative examples depending on split:
            - train: 1 positive + 1 corrupted negative
            - dev: use given labels if present, otherwise generate negative
            - test: use given labels if present, otherwise create 40 corruptions
                (20 head + 20 tail) for ranking evaluation

            Parameters:

            lines:
                Raw TSV rows. Expected triple format:
                [head_id, relation_id, tail_id, optional_label]
            set_type:
                Dataset split name: "train", "dev", or "test".
            data_dir:
                Directory containing entity/relation text mapping files.

            Returns
            -------
            List[InputExample]
                Prepared examples for the requested split.
        """
        ent2text = self._load_ent2text(data_dir)
        rel2text = self._load_rel2text(data_dir)
        entities = list(ent2text.keys())

        # Keep a fast lookup set of known triples to avoid generating false negatives
        lines_str_set = set(['\t'.join(line[:3]) for line in lines if len(line) >= 3])
        examples = []

        for i, line in enumerate(tqdm(lines, desc=f"Building {set_type} examples")):
            # Skip malformed rows
            if len(line) < 3:
                continue
            # Skip triples whose ids are missing in mapping files
            if line[0] not in ent2text or line[2] not in ent2text or line[1] not in rel2text:
                continue
            # Convert triple ids into natural-language text
            text_a = ent2text[line[0]]
            text_b = rel2text[line[1]]
            text_c = ent2text[line[2]]
            guid = "%s-%s" % (set_type, i)

            if set_type == "train":
                # Positive triple
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label="1"))
                # Add one negative sample by corrupting head or tail
                self.corrupt_head_tail(ent2text, entities, examples, i, line, lines_str_set, set_type, text_a, text_b, text_c)
            elif set_type == "dev":
                # If label is already provided in file, use it directly
                if len(line) >= 4 and line[3] in ["0", "1"]:
                    label = line[3]
                    self.labels.add(label)
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
                
                else:
                    # Otherwise treat triple as positive and generate one negative
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label="1"))
                    self.corrupt_head_tail(ent2text, entities, examples, i, line, lines_str_set, set_type, text_a, text_b, text_c)
            elif set_type == "test":
                # If label exists, use it directly
                if len(line) >= 4 and line[3] in ["0", "1"]:
                    label = line[3]
                    self.labels.add(label)
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
                else:
                    # Ranking-style evaluation:
                    # 1 positive + 20 corrupted heads + 20 corrupted tails = 41 candidates
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label="1"))
                    self.corrupt_all_head_tail(ent2text, entities, examples, i, line, set_type, text_a, text_b, text_c, corrupt_items=20)
        return examples


def example_to_text(example):
    """
        Convert a triple example into the text format expected by the encoder.

        Format:
        
        classification: [HEAD] ... [REL] ... [TAIL] ...

        This structured prefix helps the embedding model understand
        the role of each triple component.

        Parameters:
        
        example:
            An `InputExample` containing head, relation, and tail text.

        Returns:
        
        str
            Serialized triple text for tokenization.
    """
    return f"classification: [HEAD] {example.text_a} [REL] {example.text_b} [TAIL] {example.text_c}"


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info=True):
    """
    Tokenize examples and convert them into model-ready features.

    Processing Steps:
    
    1) Convert label strings into numeric ids.
    2) Serialize each triple into a single text sequence.
    3) Tokenize with truncation/padding to fixed length.
    4) Store tensors needed for model input.

    Parameters:
    
    examples:
        List of `InputExample` objects.
    label_list:
        List of allowed label strings, e.g. ["0", "1"].
    max_seq_length:
        Maximum sequence length for tokenizer output.
    tokenizer:
        Hugging Face tokenizer used for encoding.
    print_info:
        If True, log a few sample examples for debugging.

    Returns:
    
    List[InputFeatures]
        Tokenized examples with input ids, attention mask, and label id.
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(tqdm(examples, desc="Tokenizing examples")):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        # Convert structured triple into a single input string
        text = example_to_text(example)

        # Tokenize and pad/truncate to fixed maximum length
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )

        # Convert string label to integer id
        label_id = label_map[example.label]

        # Print a few early examples for sanity checking
        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("text: %s", text)
            logger.info("label: %s (id = %d)", example.label, label_id)

        features.append(
            InputFeatures(
                input_ids=encoded["input_ids"].squeeze(0),
                attention_mask=encoded["attention_mask"].squeeze(0),
                label_id=label_id,
            )
        )
    return features


def create_tensor_dataset(features):
    """
    Convert feature objects into a PyTorch `TensorDataset`.

    Parameters:

    features:
        List of `InputFeatures`.

    Returns:

    TensorDataset
        Dataset containing:
        - input_ids
        - attention_mask
        - label_ids
    """
    all_input_ids = torch.stack([f.input_ids for f in features])
    all_attention_mask = torch.stack([f.attention_mask for f in features])
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_attention_mask, all_label_ids)


def mean_pooling(model_output, attention_mask):
    """
    Apply mean pooling over token embeddings using the attention mask.

    Why this is needed:
    
    The encoder outputs one embedding per token. For classification,
    we need one fixed-size vector per input sequence. Mean pooling
    averages only the valid (non-padding) token embeddings.

    Parameters:

    model_output:
        Output tuple from the transformer encoder.
    attention_mask:
        Mask indicating real tokens (1) vs padding (0).

    Returns:
    
    torch.Tensor
        Sequence-level pooled embeddings of shape `(batch_size, hidden_size)`.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class NomicTripleClassifier(nn.Module):

    """
        Transformer-based binary classifier for KG triple classification.

        Model Architecture:
        
        1) Encode the serialized triple text using a pretrained embedding model.
        2) Apply mean pooling over token embeddings to obtain one vector per triple.
        3) Apply dropout for regularization.
        4) Use a linear classification head to predict:
        - class 0: corrupted / invalid triple
        - class 1: valid / positive triple

        Parameters:
        
        model_name:
            Hugging Face model id/path for the encoder backbone.
        num_labels:
            Number of output classes. For this task, it is 2`.

    """
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        # Load encoder backbone (e.g. nomic-ai/nomic-embed-text-v1.5)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # Hidden size of the encoder output
        hidden_size = self.encoder.config.hidden_size
        # Light regularization before classification
        self.dropout = nn.Dropout(0.1)
        # Final linear layer for binary classification logits
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Run a forward pass and return classification logits.

        Parameters:
        
        input_ids:
            Token ids of shape `(batch_size, seq_len)`.
        attention_mask:
            Attention mask of shape `(batch_size, seq_len)`.

        Returns:
        
        torch.Tensor
            Logits of shape `(batch_size, num_labels)`.
        """
        # Encode token sequence
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Convert token-level embeddings -> sequence-level embedding
        pooled = mean_pooling(outputs, attention_mask)
        # Apply dropout + classification head
        logits = self.classifier(self.dropout(pooled))
        return logits


def simple_accuracy(preds, labels):
    """
    Compute standard classification accuracy.

    Parameters:
    
    preds:
        Predicted label ids.
    labels:
        Gold label ids.

    Returns:
   
    float
        Mean accuracy over all examples.
    """
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    """
        Compute evaluation metrics for the requested task.

        Currently Supported
        -------------------
        - kg: binary classification accuracy

        Parameters
        ----------
        task_name:
            Name of the task.
        preds:
            Predicted label ids.
        labels:
            Gold label ids.

        Returns
        -------
        Dict[str, float]
            Metric dictionary.

        Raises
        ------
        KeyError
            If task is unsupported.
    """
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    raise KeyError(task_name)


def tc_compute_metrics(logits, group_size=41):
    """
    Compute ranking-based triple classification metrics.

    Evaluation Setup:
    
    Test examples are arranged in groups of 41:
    - 1 positive triple
    - 20 head-corrupted negatives
    - 20 tail-corrupted negatives

    The positive triple is assumed to be at index 0 inside each group.
    We rank all 41 candidates by the score for class 1 (valid triple).

    Metrics:
    
    - Mean Reciprocal Rank (MRR)
    - Hits@1
    - Hits@3
    - Hits@5
    - Hits@10

    Parameters:
    
    logits:
        Model output logits for all test examples.
        Shape: `(num_examples, 2)`
    group_size:
        Number of candidates per ranking group. Default is 41.

    Returns:
    
    Dict[str, float]
        Ranking metric values.
    """
    ranks = []
    # Store hit results for Hits@1 ... Hits@10
    hits = [[] for _ in range(10)]

    for triple_id in range(0, len(logits), group_size):
        # Extract class-1 scores (probability/logit of being a valid triple)
        group_preds = logits[triple_id:triple_id + group_size, 1]

        # Skip incomplete groups
        if len(group_preds) < group_size:
            continue
        rel_values = torch.tensor(group_preds)
        # Sort candidates by descending positive-class score
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        # Positive triple is placed at index 0 in each group
        rank = np.where(argsort1 == 0)[0][0]
        ranks.append(rank + 1)

        # Record hit indicators
        for hits_level in range(10):
            hits[hits_level].append(1.0 if rank <= hits_level else 0.0)

    metrics_with_values = {
        'mean_reciprocal_rank': np.mean(1. / np.array(ranks)) if ranks else 0.0,
    }
    # Report the most commonly used Hits@K values
    for i in [0, 2, 4, 9]:
        metrics_with_values[f'hits_@{i + 1}'] = np.mean(hits[i]) if hits[i] else 0.0
    return metrics_with_values


def evaluate(model, device, dataloader, num_labels):
    """
    Run evaluation or prediction on a dataloader.

    Processing Flow:
    1) Switch model to eval mode.
    2) Disable gradients.
    3) Run forward pass batch by batch.
    4) Compute average cross-entropy loss.
    5) Collect raw logits, predicted labels, and gold labels.

    Parameters:
    model:
        Triple classification model.
    device:
        Torch device ("cpu" or "cuda").
    dataloader:
        Evaluation/test dataloader.
    num_labels:
        Number of output classes.

    Returns:
    Tuple[float, np.ndarray, np.ndarray, np.ndarray]
        - eval_loss: average loss
        - preds: raw logits for all examples
        - pred_labels: argmax class predictions
        - gold: gold labels
    """
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    gold = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch tensors to target device
            input_ids, attention_mask, label_ids = (t.to(device) for t in batch)
            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            # Compute batch loss
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            # Store predictions and labels for later metric computation
            preds.append(logits.detach().cpu().numpy())
            gold.append(label_ids.detach().cpu().numpy())
    # Average loss across evaluation steps
    eval_loss = eval_loss / max(nb_eval_steps, 1)
    # Merge all batches into full arrays
    preds = np.concatenate(preds, axis=0) if preds else np.zeros((0, num_labels))
    gold = np.concatenate(gold, axis=0) if gold else np.zeros((0,), dtype=np.int64)
    # Standard classification predictions
    pred_labels = np.argmax(preds, axis=1) if len(preds) else np.array([])
    return eval_loss, preds, pred_labels, gold


def main():

    """
    Entry point for training, evaluation, and prediction.

    Workflow:
    1) Parse command-line arguments.
    2) Set device, logging, and random seeds.
    3) Load dataset processor, tokenizer, and model.
    4) Optionally train the classifier.
    5) Optionally evaluate on dev set.
    6) Optionally predict on test set and compute ranking metrics.

    Supported Task"
    - kg : binary triple classification with ranking-style test evaluation
    """
    parser = argparse.ArgumentParser()
    # Dataset / model / output paths
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Use nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # Sequence / run configuration
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    # Optimization settings
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    # Runtime settings
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Select GPU if available and not explicitly disabled
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    # Configure logging format for easier debugging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info("device: %s n_gpu: %s, distributed training: False", device, n_gpu)
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # Ensure at least one run mode is selected

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of do_train/do_eval/do_predict must be True.")
    
    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # This script currently supports only the KG triple classification task
    task_name = args.task_name.lower()
    if task_name != "kg":
        raise ValueError("Task not found: %s" % task_name)
    
    # Prepare processor and label space
    processor = KGProcessor()
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)

    # Load tokenizer and classification model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)
    model = NomicTripleClassifier(args.bert_model, num_labels=num_labels)
    model.to(device)

    # Optimizer and training loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    # Training bookkeeping
    global_step = 0
    tr_loss = 0.0
    nb_tr_steps = 0

    if args.do_train:
        """
        Training Phase
        --------------
        - Load train examples
        - Tokenize and build DataLoader
        - Train for the requested number of epochs
        - Save model weights
        """
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        train_data = create_tensor_dataset(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epochs = %s", args.num_train_epochs)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in tqdm(train_dataloader, desc="Iteration"):

                # Move batch to device
                input_ids, attention_mask, label_ids = (t.to(device) for t in batch)
                # Forward pass
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                # Compute classification loss
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                # Backpropagation + optimizer step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Track losses and steps
                tr_loss += loss.item()
                nb_tr_steps += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

            logger.info("Training loss: %.6f", epoch_loss / max(epoch_steps, 1))
        # Save trained model weights
        torch.save(model.state_dict(), os.path.join(args.output_dir, "nomic_tc_model.pt"))
        logger.info("Saved model to %s", os.path.join(args.output_dir, "nomic_tc_model.pt"))

    elif os.path.exists(os.path.join(args.output_dir, "nomic_tc_model.pt")):
        """
        Model Loading Phase
        -------------------
        If training is skipped but a saved model exists, load it for eval/predict.
        """
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "nomic_tc_model.pt"), map_location=device))
        model.to(device)

    if args.do_eval:
        """
        Development Evaluation Phase

        - Load dev examples
        - Run evaluation
        - Compute classification accuracy
        - Save results to eval_results.txt
        """
        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_features = convert_examples_to_features(dev_examples, label_list, args.max_seq_length, tokenizer)
        dev_data = create_tensor_dataset(dev_features)
        dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss, raw_preds, pred_labels, gold = evaluate(model, device, dev_dataloader, num_labels)
        result = compute_metrics(task_name, pred_labels, gold)
        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = (tr_loss / nb_tr_steps) if nb_tr_steps > 0 else None

        with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_predict:
        """
        Test Prediction Phase
        
        - Load test examples
        - Run forward pass on all candidates
        - Compute ranking metrics (MRR, Hits@K)
        - Save results to test_results.txt
        """
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        test_data = create_tensor_dataset(test_features)
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.eval_batch_size)

        logger.info("***** Running Prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_loss, raw_preds, pred_labels, gold = evaluate(model, device, test_dataloader, num_labels)
        # Ranking metrics are computed over grouped candidates (default: 41)
        result = tc_compute_metrics(raw_preds, group_size=41)
        result['eval_loss'] = test_loss
        result['global_step'] = global_step
        result['loss'] = (tr_loss / nb_tr_steps) if nb_tr_steps > 0 else None

        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
