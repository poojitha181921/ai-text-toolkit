import time
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from toolkit.schemas import ToolkitResponse, ProcessingMetadata, SummaryData

class TextSummarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        
        # Use GPU (cuda) if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and seq2seq model directly (compatible with Transformers v5+)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def summarize(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30
    ) -> ToolkitResponse:
        # 1. Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to(self.device)
        
        input_tokens = inputs["input_ids"].shape[1]
        original_char_count = len(text)

        # 2. Measure latency during inference
        start_time = time.perf_counter()
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        # 3. Decode output tokens
        summary_text = self.tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True
        ).strip()
        
        summary_char_count = len(summary_text)
        compression_ratio = round(summary_char_count / max(original_char_count, 1), 3)
        output_tokens = summary_ids.shape[1]

        # 4. Structure data through Pydantic
        summary_data = SummaryData(
            summary_text=summary_text,
            original_char_count=original_char_count,
            summary_char_count=summary_char_count,
            compression_ratio=compression_ratio
        )

        metadata = ProcessingMetadata(
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_token_count=input_tokens,
            output_token_count=output_tokens
        )

        return ToolkitResponse(
            task="summarization",
            status="success",
            data=summary_data.model_dump(),
            metadata=metadata
        )