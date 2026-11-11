import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from toolkit.classifier import TextClassifier
from toolkit.summarizer import TextSummarizer
from evaluation.metrics import (
    compute_latency_stats,
    evaluate_classification_predictions,
    compute_rouge_scores
)

console = Console()

def run_benchmarks():
    dataset_path = Path("data/eval_dataset.json")
    if not dataset_path.exists():
        console.print("[bold red]Dataset not found at data/eval_dataset.json[/bold red]")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # ----------------------------------------------------
    # 1. Benchmark Classification
    # ----------------------------------------------------
    console.print("\n[bold cyan]1. Evaluating Text Classifier...[/bold cyan]")
    classifier = TextClassifier()
    clf_tests = eval_data.get("classification_tests", [])

    y_true = []
    y_pred = []
    clf_latencies = []

    for item in clf_tests:
        res = classifier.classify(item["text"], candidate_labels=item["candidate_labels"])
        predicted = res.data["top_label"]
        y_true.append(item["expected_label"])
        y_pred.append(predicted)
        clf_latencies.append(res.metadata.latency_ms)

    clf_metrics = evaluate_classification_predictions(y_true, y_pred)
    clf_latency_stats = compute_latency_stats(clf_latencies)

    # Display Classification Results
    clf_table = Table(title="Classification Benchmark")
    clf_table.add_column("Metric", style="bold")
    clf_table.add_column("Value", style="green")

    for k, v in clf_metrics.items():
        clf_table.add_row(k.replace("_", " ").title(), str(v))
    for k, v in clf_latency_stats.items():
        clf_table.add_row(f"Latency {k}", f"{v} ms" if "ms" in k else str(v))

    console.print(clf_table)

    # ----------------------------------------------------
    # 2. Benchmark Summarization
    # ----------------------------------------------------
    console.print("\n[bold cyan]2. Evaluating Summarizer...[/bold cyan]")
    summarizer = TextSummarizer()
    sum_tests = eval_data.get("summarization_tests", [])

    generated = []
    references = []
    sum_latencies = []

    for item in sum_tests:
        res = summarizer.summarize(item["text"])
        generated.append(res.data["summary_text"])
        references.append(item["reference_summary"])
        sum_latencies.append(res.metadata.latency_ms)

    rouge_metrics = compute_rouge_scores(generated, references)
    sum_latency_stats = compute_latency_stats(sum_latencies)

    # Display Summarization Results
    sum_table = Table(title="Summarization Benchmark")
    sum_table.add_column("Metric", style="bold")
    sum_table.add_column("Value", style="green")

    for k, v in rouge_metrics.items():
        sum_table.add_row(k.upper(), str(v))
    for k, v in sum_latency_stats.items():
        sum_table.add_row(f"Latency {k}", f"{v} ms" if "ms" in k else str(v))

    console.print(sum_table)

    # Export benchmark report to JSON
    report = {
        "classification": {**clf_metrics, "latency": clf_latency_stats},
        "summarization": {**rouge_metrics, "latency": sum_latency_stats}
    }
    report_path = Path("evaluation/benchmark_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"\n[green]Benchmark report successfully saved to {report_path.resolve()}[/green]")

if __name__ == "__main__":
    run_benchmarks()