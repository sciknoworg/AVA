from transformers import TrainerCallback
import json
import numpy as np
import torch
from datasets import Dataset
import random
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
from .hyperbolic import HyperbolicTripletLoss
from sentence_transformers.sentence_transformer import losses
from .dpo import EmbeddingDPOLoss

class MemoryCleanupCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def on_save(self, args, state, control, **kwargs):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

class WeightedLoss(torch.nn.Module):
    def __init__(self, base_loss: torch.nn.Module, weight: float):
        super().__init__()
        self.base_loss = base_loss
        self.weight = weight
    def forward(self, *args, **kwargs):
        return self.weight * self.base_loss(*args, **kwargs)

def set_seed(seed: int) -> None:
    """
    Set the global random seed for Python, NumPy, and PyTorch to ensure
    reproducible training runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(records: list[dict]) -> Dataset:
    """
    Convert a list of triplet dicts into a HuggingFace ``Dataset``.
    """
    return Dataset.from_dict({
        "anchor":   [r["anchor"]   for r in records],
        "positive": [r["positive"] for r in records],
        "negative": [r["negative"] for r in records],
    })


def make_evaluator(eval_dataset, eval_bs=32):
    triplet_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="triplet-eval-full",
        show_progress_bar=True,
        batch_size=eval_bs,
    )
    return triplet_evaluator


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def make_losses(model, ref_model=None,
                use_hyperbolic=False,
                use_dpo = False,
                rl_beta=0.1,
                triplet_margin=0.3,
                c=0.1,
                w_hyperbolic=1.0,
                w_dpo=1.0):
    if use_hyperbolic and use_dpo:
        triplet_loss_module = HyperbolicTripletLoss(model, margin=triplet_margin, c=c)
        return {
            "hyperbolic": WeightedLoss(triplet_loss_module, w_hyperbolic),
            "dpo": WeightedLoss(EmbeddingDPOLoss(model, ref_model, beta=rl_beta), w_dpo),
        }
    if use_hyperbolic:
        triplet_loss_module = HyperbolicTripletLoss(model, margin=triplet_margin, c=c)
        print(f"Using HyperbolicTripletLoss with margin={triplet_margin} and c={c}")
        return triplet_loss_module
    elif use_dpo:
        if ref_model is None:
            print("REFERENCE model is None! DPO loss cannot be used without a reference model.")
        return EmbeddingDPOLoss(model, ref_model, beta=rl_beta)
    else:
        return losses.TripletLoss(
            model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=triplet_margin,
        )

