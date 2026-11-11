import json
from pathlib import Path
from toolkit.rag import SimpleRAGPipeline

if __name__ == "__main__":
    doc = Path("data/sample_knowledge.txt").read_text(encoding="utf-8")
    
    print("Initializing and indexing RAG pipeline...")
    rag = SimpleRAGPipeline()
    rag.index_documents([doc], chunk_size=40)
    
    question = "Who was the second person to walk on the Moon?"
    print(f"Querying: {question}")
    
    response = rag.query(question)
    print("\n--- Structured RAG Output ---")
    print(json.dumps(response.model_dump(), indent=2))