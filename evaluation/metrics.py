from typing import List, Dict, Any
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculates Mean, Min, Max, P50, and P95 latency percentiles."""
    if not latencies_ms:
        return {}
    return {
        "count": len(latencies_ms),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "min_ms": round(float(np.min(latencies_ms)), 2),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "max_ms": round(float(np.max(latencies_ms)), 2),
    }

def evaluate_classification_predictions(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, Any]:
    """Computes categorical accuracy and weighted Precision, Recall, and F1."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1": round(float(f1), 4)
    }

def compute_rouge_scores(
    generated_summaries: List[str],
    reference_summaries: List[str]
) -> Dict[str, float]:
    """Computes standard ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []

    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = scorer.score(ref, gen)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1_f1": round(float(np.mean(r1)), 4),
        "rouge2_f1": round(float(np.mean(r2)), 4),
        "rougeL_f1": round(float(np.mean(rl)), 4)
    }