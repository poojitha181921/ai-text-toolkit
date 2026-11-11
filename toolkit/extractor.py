import time
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from toolkit.schemas import ToolkitResponse, ProcessingMetadata, ExtractionData, EntityItem

class InfoExtractor:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name).to(self.device)
        self.id2label = self.model.config.id2label

    def extract_entities(self, text: str) -> ToolkitResponse:
        start_time = time.perf_counter()

        # Tokenize with character offset mapping to accurately recover word spans
        inputs = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        input_tokens = inputs["input_ids"].shape[1]
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        predictions = torch.argmax(probabilities, dim=-1).tolist()

        raw_entities = []
        for idx, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
            label = self.id2label[pred_id]
            # Ignore special tokens ([CLS], [SEP]) and non-entities ('O')
            if label == "O" or offset == [0, 0]:
                continue

            score = round(probabilities[idx][pred_id].item(), 4)
            raw_entities.append({
                "token_id": idx,
                "label": label,
                "score": score,
                "start": offset[0],
                "end": offset[1],
                "text": text[offset[0]:offset[1]]
            })

        # Merge subword tokens (e.g. B-PER followed by I-PER) into complete entities
        merged_entities: List[EntityItem] = []
        for item in raw_entities:
            tag_type = item["label"].split("-")[-1]  # PER, ORG, LOC, MISC
            is_start = item["label"].startswith("B-")

            if merged_entities and not is_start and merged_entities[-1].label == tag_type and item["start"] <= merged_entities[-1].end + 1:
                # Extend the prior entity span
                prev = merged_entities[-1]
                prev.end = item["end"]
                prev.text = text[prev.start:prev.end]
                prev.score = round(min(prev.score, item["score"]), 4)
            else:
                merged_entities.append(
                    EntityItem(
                        text=item["text"],
                        label=tag_type,
                        score=item["score"],
                        start=item["start"],
                        end=item["end"]
                    )
                )

        # Build grouped dictionary for structured extraction: {"ORG": [...], "PER": [...]}
        grouped: Dict[str, List[str]] = {}
        for ent in merged_entities:
            grouped.setdefault(ent.label, [])
            if ent.text not in grouped[ent.label]:
                grouped[ent.label].append(ent.text)

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        extraction_data = ExtractionData(
            text=text,
            entities=merged_entities,
            grouped_entities=grouped
        )

        metadata = ProcessingMetadata(
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_token_count=input_tokens,
            output_token_count=None
        )

        return ToolkitResponse(
            task="information_extraction",
            status="success",
            data=extraction_data.model_dump(),
            metadata=metadata
        )