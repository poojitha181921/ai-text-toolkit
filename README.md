# AI Text Processing Toolkit (Prototype)

A modular, production-ready NLP toolkit providing structured inference pipelines, CLI interfaces, evaluation metrics, and local RAG workflows.

## Features
- **Abstractive Summarization:** Hugging Face Seq2Seq models with compression ratio and token tracking.
- **Zero-Shot Classification:** NLI-based categorization with multi-class and multi-label probability distributions.
- **Information Extraction:** Named Entity Recognition (NER) with token offset alignment and entity grouping.
- **Strict JSON Schemas:** Validated end-to-end via Pydantic with execution metadata and latency profiling.
- **CLI Suite:** Interactive terminal interface built with Typer and Rich, including file logging.
- **Evaluation Suite:** Automated benchmarking reporting accuracy, precision, recall, F1, ROUGE-1/2/L, and latency percentiles (P50, P95).
- **Local RAG Pipeline:** Semantic chunking, dense vector indexing via Sentence-Transformers, and grounded generation with Flan-T5.

## Installation
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt