import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from typing import Any
from tqdm import tqdm


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    Row-wise L2 normalisation of a 2-D embedding matrix.

    Each row is divided by its Euclidean norm.  A small epsilon (1e-8) is
    added to the denominator to guard against division by zero for zero
    vectors.

    Parameters
    ----------
    x : np.ndarray, shape (N, D)
        Matrix of N embeddings, each of dimensionality D.

    Returns
    -------
    np.ndarray, shape (N, D)
        Unit-norm embeddings.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-8)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pair-wise cosine similarity between two embedding matrices.

    Normalises both inputs before computing the dot product, so the result
    is always in [-1, 1] regardless of whether the model already outputs
    unit vectors.

    Parameters
    ----------
    a : np.ndarray, shape (N, D)
    b : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N,)
        Cosine similarity score for each of the N pairs.
    """
    return np.sum(_l2_normalize(a) * _l2_normalize(b), axis=1)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def triplet_accuracy(
    a_emb: np.ndarray,
    p_emb: np.ndarray,
    n_emb: np.ndarray,
) -> float:
    """
    Fraction of triplets where the anchor is closer to the positive than to
    the negative (strict inequality).

    This is the most direct measure of whether the model respects the
    intended ordering: sim(anchor, positive) > sim(anchor, negative).

    Parameters
    ----------
    a_emb : np.ndarray, shape (N, D)
        Anchor embeddings.
    p_emb : np.ndarray, shape (N, D)
        Positive embeddings.
    n_emb : np.ndarray, shape (N, D)
        Negative embeddings.

    Returns
    -------
    float
        Proportion of correctly ordered triplets in [0, 1].
        1.0 means the model perfectly separates all positives from negatives.

    Notes
    -----
    Ties (sim_ap == sim_an) are counted as *incorrect*, which is the
    conservative convention used in most embedding benchmarks.
    """
    sim_ap = _cosine_similarity(a_emb, p_emb)
    sim_an = _cosine_similarity(a_emb, n_emb)
    return float(np.mean(sim_ap > sim_an))


def retrieval_eval(
    a_emb: np.ndarray,
    p_emb: np.ndarray,
    n_emb: np.ndarray,
) -> dict[str, float]:
    """
    Retrieval-style evaluation assuming a two-candidate pool {positive, negative}.

    For each anchor the model ranks the positive and the negative by cosine
    similarity.  Because the pool has exactly two candidates, only two rank
    positions are possible (1 and 2).

    Metrics computed
    ----------------
    MRR (Mean Reciprocal Rank)
        Average of 1 / rank(positive).
        - Positive ranked 1st  → reciprocal rank = 1.0
        - Positive ranked 2nd  → reciprocal rank = 0.5
        - Tie                  → reciprocal rank = 2/3  (average of 1/1 and 1/2,
                                 i.e. both candidates share rank 1.5 by convention)

    Recall@1
        Fraction of queries where the positive is the top-ranked candidate.
        Equivalent to ``triplet_accuracy`` but expressed as a retrieval metric.
        Ties are *not* counted as a top-1 hit.

    Parameters
    ----------
    a_emb : np.ndarray, shape (N, D)
    p_emb : np.ndarray, shape (N, D)
    n_emb : np.ndarray, shape (N, D)

    Returns
    -------
    dict with keys:
        "MRR"       : float in (0, 1]
        "Recall@1"  : float in [0, 1]
    """
    sim_ap = _cosine_similarity(a_emb, p_emb)
    sim_an = _cosine_similarity(a_emb, n_emb)

    # For a 2-candidate pool, a tie means both share rank 1 and rank 2.
    # The expected rank of the positive is 1.5  → reciprocal rank = 2/3.
    rr = np.where(
        sim_ap > sim_an,
        1.0,
        np.where(sim_ap == sim_an, 2.0 / 3.0, 0.5),
    )

    return {
        "MRR": float(np.mean(rr)),
        "Recall@1": float(np.mean(sim_ap > sim_an)),
    }


def hard_negative_eval(
    a_emb: np.ndarray,
    p_emb: np.ndarray,
    n_emb: np.ndarray,
    data: list[dict[str, Any]],
    score_field: str = "anchor_negaive_score",
    threshold: float = 90.0,
) -> float | None:
    """
    Triplet accuracy restricted to *hard* negatives.

    Hard negatives are samples whose ``score_field`` value is at or above
    ``threshold``.  A high anchor–negative similarity score indicates that
    the negative is lexically or semantically close to the anchor, making
    it harder for the model to distinguish from the positive.

    Parameters
    ----------
    a_emb : np.ndarray, shape (N, D)
    p_emb : np.ndarray, shape (N, D)
    n_emb : np.ndarray, shape (N, D)
    data : list of dict
        Raw data records corresponding row-by-row to the embedding arrays.
        Each dict must contain the key specified by ``score_field``.
    score_field : str, optional
        Name of the field that holds the anchor–negative similarity score.
        Defaults to ``"anchor_negaive_score"`` to match the dataset schema
        (note the intentional typo preserved from the source data).
    threshold : float, optional
        Minimum score (inclusive) for a sample to be treated as a hard
        negative.  Defaults to 90.0.

    Returns
    -------
    float or None
        Triplet accuracy on the hard-negative subset, in [0, 1].
        Returns ``None`` (not 0.0) when no samples satisfy the threshold,
        so callers can distinguish "perfect failure" from "no data".

    Raises
    ------
    KeyError
        If ``score_field`` is not present in a data record.
    """
    valid_idxs = [
        i for i, item in enumerate(data)
        if item[score_field] >= threshold
    ]

    if not valid_idxs:
        return None  # Distinguish "no qualifying samples" from a real score

    a_v = a_emb[valid_idxs]
    p_v = p_emb[valid_idxs]
    n_v = n_emb[valid_idxs]

    sim_ap = _cosine_similarity(a_v, p_v)
    sim_an = _cosine_similarity(a_v, n_v)
    return float(np.mean(sim_ap > sim_an))


def classification_metrics(
    a_emb: np.ndarray,
    p_emb: np.ndarray,
    n_emb: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """
    Binary classification metrics by treating similarity scoring as a
    pair-level classifier.

    All (anchor, positive) pairs are labelled 1 (match) and all
    (anchor, negative) pairs are labelled 0 (non-match). The cosine
    similarity score acts as the classifier's predicted probability.

    If `threshold` is None, we dynamically find the optimal threshold that
    maximizes the F1 score. Otherwise, we evaluate using the provided threshold.

    Parameters
    ----------
    a_emb : np.ndarray, shape (N, D)
    p_emb : np.ndarray, shape (N, D)
    n_emb : np.ndarray, shape (N, D)
    threshold : float, optional

    Returns
    -------
    dict with keys:
        "accuracy"      : fraction of correctly classified pairs at best F1 threshold
        "precision"     : TP / (TP + FP) at best threshold
        "recall"        : TP / (TP + FN) at best threshold
        "f1"            : harmonic mean of precision and recall (maximized)
        "best_threshold": the similarity threshold that maximized F1
        "roc_auc"       : area under the ROC curve (threshold-independent)
        "avg_precision" : area under the precision–recall curve (threshold-independent)
    """
    sim_ap = _cosine_similarity(a_emb, p_emb)
    sim_an = _cosine_similarity(a_emb, n_emb)

    y_true = np.concatenate([np.ones(len(a_emb)), np.zeros(len(a_emb))])
    y_scores = np.concatenate([sim_ap, sim_an])

    if threshold is None:
        # Find the ideal threshold utilizing the Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        best_threshold = float(threshold)

    y_pred = (y_scores >= best_threshold).astype(int)

    return {
        "best_threshold": float(best_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "avg_precision": float(average_precision_score(y_true, y_scores)),
    }


def openai_encode(texts, client, model_name, batch_size=1000):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    embedding_matrix = np.array(all_embeddings)
    return embedding_matrix

def run_evaluation_report(
    model: Any,
    data: list[dict[str, Any]],
    batch_size: int = 64,
    classification_threshold: float | None = None,
    hard_negative_threshold: float = 90.0,
    hard_negative_score_field: str = "anchor_negaive_score",
    is_openai_model: bool = False,
    query_prefix: str = "",
    doc_prefix: str = "",
) -> dict[str, float | None]:
    """
    Encode a triplet dataset with ``model`` and compute a full evaluation
    report covering triplet accuracy, retrieval quality, hard-negative
    discrimination, and classification performance.

    Parameters
    ----------
    model : any object with an ``encode`` method
        A SentenceTransformer-compatible model.  The ``encode`` method must
        accept a list of strings and keyword arguments ``batch_size``,
        ``convert_to_numpy``, and ``show_progress_bar``.
    data : list of dict
        Triplet records.  Each dict must contain at minimum the keys
        ``"anchor"``, ``"positive"``, and ``"negative"``.  Hard-negative
        evaluation additionally requires the field named by
        ``hard_negative_score_field``.
    batch_size : int, optional
        Number of sentences per encoding batch.  Defaults to 64.
    classification_threshold : float, optional
        Fixed cosine-similarity decision boundary for the binary
        classification metrics.  Defaults to 0.5.
    hard_negative_threshold : float, optional
        Minimum anchor–negative score for a sample to be counted as a hard
        negative.  Defaults to 90.0.
    hard_negative_score_field : str, optional
        Data field holding the anchor–negative similarity score used to
        filter hard negatives.  Defaults to ``"anchor_negaive_score"``.

    Returns
    -------
    dict
        Flat mapping of metric name → value.  All values are ``float``
        except ``hard_negative_acc``, which may be ``None`` when no samples
        satisfy the hard-negative threshold.

    Metrics returned
    ----------------
    triplet_accuracy    Fraction of triplets correctly ordered by cosine sim.
    MRR                 Mean Reciprocal Rank over the two-candidate pool.
    Recall@1            Fraction of queries where positive is top-ranked.
    hard_negative_acc   Triplet accuracy on hard-negative subset (or None).
    threshold           Decision boundary used for classification metrics.
    accuracy            Binary pair-classification accuracy.
    precision           Precision at the chosen threshold.
    recall              Recall at the chosen threshold.
    f1                  F1-score at the chosen threshold.
    roc_auc             ROC-AUC (threshold-independent).
    avg_precision       Average Precision / PR-AUC (threshold-independent).
    """
    print(f"\n{'=' * 40}")
    print(f"Evaluation  |  samples: {len(data)}")
    print(f"{'=' * 40}")

    anchors   = [query_prefix + x["anchor"]   for x in data]
    positives = [doc_prefix + x["positive"] for x in data]
    negatives = [doc_prefix + x["negative"] for x in data]

    if is_openai_model:
        from openai import OpenAI
        from dotenv import find_dotenv, load_dotenv
        import os
        _ = load_dotenv(find_dotenv())
        client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
        print("\nEncoding anchors …")
        a_emb = openai_encode(anchors, client, model)
        print("Encoding positives …")
        p_emb = openai_encode(positives, client, model)
        print("Encoding negatives …")
        n_emb = openai_encode(negatives, client, model)
    else:
        encode_kwargs = dict(
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        print("\nEncoding anchors …")
        a_emb = model.encode(anchors, **encode_kwargs)
        print("Encoding positives …")
        p_emb = model.encode(positives, **encode_kwargs)
        print("Encoding negatives …")
        n_emb = model.encode(negatives, **encode_kwargs)

    metrics = {}

    metrics["triplet_accuracy"] = triplet_accuracy(a_emb, p_emb, n_emb)

    metrics.update(retrieval_eval(a_emb, p_emb, n_emb))

    metrics["hard_negative_acc"] = hard_negative_eval(
        a_emb, p_emb, n_emb,
        data,
        score_field=hard_negative_score_field,
        threshold=hard_negative_threshold,
    )

    metrics.update(
        classification_metrics(
            a_emb, p_emb, n_emb,
            threshold=classification_threshold
        )
    )

    print(f"\n{'─' * 40}")
    print(f"{'Metric':<25} {'Value':>10}")
    print(f"{'─' * 40}")
    for key, val in metrics.items():
        val_str = f"{val:.4f}" if isinstance(val, float) else "N/A"
        print(f"{key:<25} {val_str:>10}")
    print(f"{'─' * 40}\n")

    return metrics