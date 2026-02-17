import re
from transformers import pipeline

# Zero-shot classifier for gap type
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

GAP_LABELS = [
    "Mathematical reasoning difficulty",
    "Conceptual misunderstanding",
    "Implementation problem",
    "Formula confusion",
    "Application confusion"
]


def extract_concepts(text):
    """
    Extract 'understood' and 'confused' concepts using rule-based parsing.
    """
    understood = None
    confused = None

    # Pattern: "I understand X but not Y"
    pattern = r"I understand (.*?) but not (.*)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        understood = match.group(1).strip()
        confused = match.group(2).strip()

    return understood, confused


def classify_gap(text):
    """
    Classify type of knowledge gap.
    """
    result = classifier(text, GAP_LABELS)
    return result["labels"][0], round(result["scores"][0], 2)
