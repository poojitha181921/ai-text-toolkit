import json
from toolkit.summarizer import TextSummarizer

if __name__ == "__main__":
    print("Loading summarizer model...")
    summarizer = TextSummarizer()

    sample_article = """
    Artificial intelligence is transforming industries worldwide, from healthcare and finance 
    to education and transportation. Natural Language Processing (NLP), a key subfield of AI, 
    enables computers to understand, interpret, and generate human language in valuable ways. 
    Modern architectures like Transformers have drastically improved the ability of models 
    to handle complex semantic relationships and long-term context across documents, paving the 
    way for intelligent personal assistants, real-time language translation, and automated document 
    summarization tools.
    """

    print("Running summarization...")
    response = summarizer.summarize(sample_article)
    
    # Print formatted JSON output
    print("\n--- Structured JSON Output ---")
    print(json.dumps(response.model_dump(), indent=2))