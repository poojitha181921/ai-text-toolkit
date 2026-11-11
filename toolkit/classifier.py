import time
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from toolkit.schemas import ToolkitResponse, ProcessingMetadata, ClassificationData, ClassificationScore

class TextClassifier:
    def __init__(self, model_name: str = "valhalla/distilbart-mnli-12-3"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    def classify(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template: str = "This text is about {}.",
        multi_label: bool = False
    ) -> ToolkitResponse:
        start_time = time.perf_counter()
        
        # Tokenize pairs: premise (text) and hypothesis (candidate label)
        pairs = [[text, hypothesis_template.format(label)] for label in candidate_labels]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1]

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # In MNLI-trained models:
        # Index 0: contradiction, Index 1: neutral (if 3-class), Index 2: entailment
        # For 2-class / 3-class MNLI, entailment is the last logit index
        entailment_id = logits.shape[-1] - 1
        entailment_logits = logits[:, entailment_id]

        if multi_label:
            # Independent probabilities per class
            probs = torch.sigmoid(entailment_logits).cpu().tolist()
        else:
            # Normalized distribution across candidate classes
            probs = torch.softmax(entailment_logits, dim=0).cpu().tolist()

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        # Pair labels with scores and sort descending
        scores = [
            ClassificationScore(label=label, score=round(prob, 4))
            for label, prob in zip(candidate_labels, probs)
        ]
        scores.sort(key=lambda x: x.score, reverse=True)

        classification_data = ClassificationData(
            text=text,
            top_label=scores[0].label,
            top_score=scores[0].score,
            all_scores=scores,
            multi_label=multi_label
        )

        metadata = ProcessingMetadata(
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_token_count=input_tokens,
            output_token_count=None
        )

        return ToolkitResponse(
            task="classification",
            status="success",
            data=classification_data.model_dump(),
            metadata=metadata
        )