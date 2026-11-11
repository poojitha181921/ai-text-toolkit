import json
from toolkit.classifier import TextClassifier

if __name__ == "__main__":
    print("Loading classifier model...")
    classifier = TextClassifier()

    sample_text = (
        "I was billed twice on my credit card for the monthly subscription. "
        "Can you please reverse the second charge immediately?"
    )
    labels = ["technical support", "billing issue", "sales inquiry", "account cancellation"]

    print("Running classification...")
    response = classifier.classify(sample_text, candidate_labels=labels)

    print("\n--- Structured Classification Output ---")
    print(json.dumps(response.model_dump(), indent=2))