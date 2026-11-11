import time
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from toolkit.schemas import ToolkitResponse, ProcessingMetadata, RAGData, RetrievedChunk


class SimpleRAGPipeline:
    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        gen_model_name: str = "google/flan-t5-base"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model_name = embed_model_name
        self.gen_model_name = gen_model_name

        # 1. Embedding Model
        self.embedder = SentenceTransformer(self.embed_model_name, device=self.device)

        # 2. Generative LM
        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(self.gen_model_name).to(self.device)

        # Internal document memory store
        self.documents: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: List[str], chunk_size: int = 150):
        """Splits documents into manageable chunks and builds vector embeddings."""
        chunks = []
        for doc in documents:
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk.strip())

        self.documents = chunks
        if self.documents:
            embeddings = self.embedder.encode(self.documents, convert_to_numpy=True, normalize_embeddings=True)
            self.doc_embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 2) -> List[RetrievedChunk]:
        """Performs dense vector retrieval via normalized cosine similarity."""
        if self.doc_embeddings is None or len(self.documents) == 0:
            return []

        query_embedding = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        # Cosine similarity for normalized vectors is simply dot product
        similarities = np.dot(self.doc_embeddings, query_embedding.T).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved = []
        for idx in top_indices:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=int(idx),
                    content=self.documents[idx],
                    similarity_score=round(float(similarities[idx]), 4)
                )
            )
        return retrieved

    def query(self, query_text: str, top_k: int = 2) -> ToolkitResponse:
        start_time = time.perf_counter()

        # Step A: Retrieve relevant context
        chunks = self.retrieve(query_text, top_k=top_k)
        context_str = "\n\n".join([f"Source [{c.chunk_id}]: {c.content}" for c in chunks])

        # Step B: Construct Grounded Prompt
        grounded_prompt = (
            f"Answer the question based strictly on the provided context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query_text}\n"
            f"Answer:"
        )

        # Step C: Generate Grounded Answer
        inputs = self.tokenizer(
            grounded_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.gen_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False
            )

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        output_tokens = output_ids.shape[1]
        answer_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        rag_data = RAGData(
            query=query_text,
            answer=answer_text,
            context_chunks=chunks,
            grounded_prompt=grounded_prompt
        )

        metadata = ProcessingMetadata(
            model_name=f"{self.embed_model_name} + {self.gen_model_name}",
            latency_ms=latency_ms,
            input_token_count=input_tokens,
            output_token_count=output_tokens
        )

        return ToolkitResponse(
            task="rag_qa",
            status="success",
            data=rag_data.model_dump(),
            metadata=metadata
        )