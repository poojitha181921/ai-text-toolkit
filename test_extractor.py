import json
from toolkit.extractor import InfoExtractor

if __name__ == "__main__":
    print("Loading entity extraction model...")
    extractor = InfoExtractor()

    sample_text = (
        "Satya Nadella, CEO of Microsoft, announced a strategic investment of $10 billion "
        "in OpenAI during an event held in Seattle, Washington."
    )

    print("Running information extraction...")
    response = extractor.extract_entities(sample_text)

    print("\n--- Structured Extraction Output ---")
    print(json.dumps(response.model_dump(), indent=2))