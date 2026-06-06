from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss
from sklearn import metrics

from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
            text_b: (Optional) string. The untokenized text of the second sequence.
            text_c: (Optional) string. The untokenized text of the third sequence.
            label: (Optional) string. The label of the example.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir
        )

    def get_relations(self, data_dir):
        with open(os.path.join(data_dir, "relations.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        return ["0", "1"]
    
    def get_entities(self, data_dir):
        with open(os.path.join(data_dir, "entities.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), "r", encoding="utf-8") as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split("\t")
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        if data_dir.find("FB15") != -1:
            long_path = os.path.join(data_dir, "entity2textlong.txt")
            if os.path.exists(long_path):
                with open(long_path, "r") as f:
                    ent_lines = f.readlines()
                    for line in ent_lines:
                        temp = line.strip().split("\t")
                        ent2text[temp[0]] = temp[1]

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), "r", encoding="utf-8") as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split("\t")
                rel2text[temp[0]] = temp[1]

        lines_str_set = set(["\t".join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":
                label = "1"
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                self.labels.add(label)
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=text_b,
                        text_c=text_c,
                        label=label,
                    )
                )

            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=text_b,
                        text_c=text_c,
                        label="1",
                    )
                )

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    for j in range(5):
                        tmp_head = ""
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = (
                                tmp_head + "\t" + line[1] + "\t" + line[2]
                            )
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_head_text = ent2text[tmp_head]
                        examples.append(
                            InputExample(
                                guid=guid,
                                text_a=tmp_head_text,
                                text_b=text_b,
                                text_c=text_c,
                                label="0",
                            )
                        )
                else:
                    tmp_tail = ""
                    for j in range(5):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + "\t" + line[1] + "\t" + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[tmp_tail]
                        examples.append(
                            InputExample(
                                guid=guid,
                                text_a=text_a,
                                text_b=text_b,
                                text_c=tmp_tail_text,
                                label="0",
                            )
                        )
        return examples


class NomicForSequenceClassification(torch.nn.Module):
    """Encoder + dropout + linear head."""

    def __init__(self, model_name_or_path, num_labels=2, trust_remote_code=True, dropout_prob=0.1):
        super(NomicForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs[0]

        pooled_output = self.mean_pool(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info=True):
    """Loads examples into InputFeatures using a single prefixed text string."""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        if example.text_b is not None and example.text_c is not None:
            text = (
                f"classification: "
                f"head: {example.text_a} "
                f"relation: {example.text_b} "
                f"tail: {example.text_c}"
            )
        else:
            text = f"classification: {example.text_a}"

        enc = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )

        input_ids = enc["input_ids"]
        input_mask = enc["attention_mask"]

        # Many encoder models either ignore token_type_ids or do not use them.
        # Keep a zero tensor for compatibility with the rest of your pipeline.
        segment_ids = enc.get("token_type_ids", [0] * len(input_ids))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("text: %s", text)
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)", example.label, label_id)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def load_model_from_checkpoint(model_name_or_path, output_dir, num_labels, device, trust_remote_code=True):
    model = NomicForSequenceClassification(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        trust_remote_code=trust_remote_code,
    )
    checkpoint_path = os.path.join(output_dir, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="Model name/path. Example: nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Unused. Kept for compatibility.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Kept for compatibility. Ignored by AutoTokenizer here.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--fp16", action="store_true",
                        help="Kept for compatibility. Not used in this version.")
    parser.add_argument("--loss_scale", type=float, default=0,
                        help="Kept for compatibility. Not used in this version.")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Pass trust_remote_code=True when loading the model.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "kg": KGProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, fp16 flag: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval`, or `do_predict` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    entity_list = processor.get_entities(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps
        ) * int(args.num_train_epochs)
        if args.local_rank != -1:
            num_train_optimization_steps = (
                num_train_optimization_steps // torch.distributed.get_world_size()
            )

    model = NomicForSequenceClassification(
        model_name_or_path=args.model_name,
        num_labels=num_labels,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = None
    if args.do_train and num_train_optimization_steps > 0:
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps,
        )

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )

        model.train()
        optimizer.zero_grad()

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                logits = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            print("Training loss: ", tr_loss, nb_tr_examples)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model_to_save = model.module if hasattr(model, "module") else model
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

        torch.save(model_to_save.state_dict(), output_model_file)
        tokenizer.save_pretrained(args.output_dir)

        model = load_model_from_checkpoint(
            model_name_or_path=args.model_name,
            output_dir=args.output_dir,
            num_labels=num_labels,
            device=device,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
            model = load_model_from_checkpoint(
                model_name_or_path=args.model_name,
                output_dir=args.output_dir,
                num_labels=num_labels,
                device=device,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            model = NomicForSequenceClassification(
                model_name_or_path=args.model_name,
                num_labels=num_labels,
                trust_remote_code=args.trust_remote_code,
            )
            model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer
        )
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
            model = load_model_from_checkpoint(
                model_name_or_path=args.model_name,
                output_dir=args.output_dir,
                num_labels=num_labels,
                device=device,
                trust_remote_code=args.trust_remote_code,
            )

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            print(label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        print("Pred label counts:", np.bincount(preds, minlength=num_labels))
        print("Gold label counts:", np.bincount(all_label_ids.numpy(), minlength=num_labels))
        print("First 20 preds vs gold:", list(zip(preds[:20], all_label_ids.numpy()[:20])))        
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result["eval_loss"] = eval_loss
        result["global_step"] = global_step
        result["loss"] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        train_triples = processor.get_train_triples(args.data_dir)
        dev_triples = processor.get_dev_triples(args.data_dir)
        test_triples = processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = "\t".join(triple)
            all_triples_str_set.add(triple_str)

        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer
        )
        logger.info("***** Running Prediction *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
            model = load_model_from_checkpoint(
                model_name_or_path=args.model_name,
                output_dir=args.output_dir,
                num_labels=num_labels,
                device=device,
                trust_remote_code=args.trust_remote_code,
            )

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        print(preds, preds.shape)

        all_label_ids = all_label_ids.numpy()
        preds = np.argmax(preds, axis=1)
        print("Pred label counts:", np.bincount(preds, minlength=num_labels))
        print("Gold label counts:", np.bincount(all_label_ids, minlength=num_labels))
        print("First 20 preds vs gold:", list(zip(preds[:20], all_label_ids[:20])))
        result = compute_metrics(task_name, preds, all_label_ids)
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result["eval_loss"] = eval_loss
        result["global_step"] = global_step
        result["loss"] = loss

        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        print("Triple classification acc is : ")
        print(metrics.accuracy_score(all_label_ids, preds))

        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])

        for test_triple in test_triples:
            head = test_triple[0]
            relation = test_triple[1]
            tail = test_triple[2]

            head_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, relation, tail]
                    tmp_triple_str = "\t".join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        head_corrupt_list.append(tmp_triple)

            tmp_examples = processor._create_examples(head_corrupt_list, "test", args.data_dir)
            tmp_features = convert_examples_to_features(
                tmp_examples, label_list, args.max_seq_length, tokenizer, print_info=False
            )
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()

            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=None,
                    )
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)

            preds = preds[0]
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            _, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()
            rank1 = np.where(argsort1 == 0)[0][0]
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)

            tail_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != tail:
                    tmp_triple = [head, relation, corrupt_ent]
                    tmp_triple_str = "\t".join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        tail_corrupt_list.append(tmp_triple)

            tmp_examples = processor._create_examples(tail_corrupt_list, "test", args.data_dir)
            tmp_features = convert_examples_to_features(
                tmp_examples, label_list, args.max_seq_length, tokenizer, print_info=False
            )
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=None,
                    )
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)

            preds = preds[0]
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            _, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()
            rank2 = np.where(argsort1 == 0)[0][0]
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            file_prefix = (
                str(args.data_dir[7:])
                + "_"
                + str(args.train_batch_size)
                + "_"
                + str(args.learning_rate)
                + "_"
                + str(args.max_seq_length)
                + "_"
                + str(args.num_train_epochs)
            )
            f = open(file_prefix + "_ranks.txt", "a")
            f.write(str(rank1) + "\t" + str(rank2) + "\n")
            f.close()

            for hits_level in range(10):
                k = hits_level + 1

                if (rank1 + 1) <= k:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if (rank2 + 1) <= k:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        logger.info("***** Link prediction results *****")
        logger.info("Hits@1: {}".format(np.mean(hits[0])))
        logger.info("Hits@3: {}".format(np.mean(hits[2])))
        logger.info("Hits@5: {}".format(np.mean(hits[4])))
        logger.info("Hits@10: {}".format(np.mean(hits[9])))
        logger.info("MRR: {}".format(np.mean(1.0 / np.array(ranks))))

        logger.info("Hits left @1: {}".format(np.mean(hits_left[0])))
        logger.info("Hits left @3: {}".format(np.mean(hits_left[2])))
        logger.info("Hits left @5: {}".format(np.mean(hits_left[4])))
        logger.info("Hits left @10: {}".format(np.mean(hits_left[9])))
        logger.info("MRR left: {}".format(np.mean(1.0 / np.array(ranks_left))))

        logger.info("Hits right @1: {}".format(np.mean(hits_right[0])))
        logger.info("Hits right @3: {}".format(np.mean(hits_right[2])))
        logger.info("Hits right @5: {}".format(np.mean(hits_right[4])))
        logger.info("Hits right @10: {}".format(np.mean(hits_right[9])))
        logger.info("MRR right: {}".format(np.mean(1.0 / np.array(ranks_right))))

        output_lp_file = os.path.join(args.output_dir, "link_prediction_metrics.txt")
        with open(output_lp_file, "w") as writer:
            writer.write("Hits@1 = {}\n".format(np.mean(hits[0])))
            writer.write("Hits@3 = {}\n".format(np.mean(hits[2])))
            writer.write("Hits@5 = {}\n".format(np.mean(hits[4])))
            writer.write("Hits@10 = {}\n".format(np.mean(hits[9])))
            writer.write("MRR = {}\n".format(np.mean(1.0 / np.array(ranks))))
            writer.write("Hits_left@1 = {}\n".format(np.mean(hits_left[0])))
            writer.write("Hits_left@3 = {}\n".format(np.mean(hits_left[2])))
            writer.write("Hits_left@5 = {}\n".format(np.mean(hits_left[4])))
            writer.write("Hits_left@10 = {}\n".format(np.mean(hits_left[9])))
            writer.write("MRR_left = {}\n".format(np.mean(1.0 / np.array(ranks_left))))
            writer.write("Hits_right@1 = {}\n".format(np.mean(hits_right[0])))
            writer.write("Hits_right@3 = {}\n".format(np.mean(hits_right[2])))
            writer.write("Hits_right@5 = {}\n".format(np.mean(hits_right[4])))
            writer.write("Hits_right@10 = {}\n".format(np.mean(hits_right[9])))
            writer.write("MRR_right = {}\n".format(np.mean(1.0 / np.array(ranks_right))))


if __name__ == "__main__":
    main()