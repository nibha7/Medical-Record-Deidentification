from hybrid_phi import regex_predictions


def check_regex_leakage(redacted_text):
    """
    Basic post-redaction scan for common structured PHI.
    This does NOT prove that a document contains no PHI:
    names, addresses, contextual identifiers and unseen formats
    require additional NER/manual review.
    """
    findings = regex_predictions(redacted_text)
    return [
        {
            "start": start,
            "end": end,
            "label": label,
            "matched_text": redacted_text[start:end]
        }
        for start, end, label, confidence in findings
    ]


if __name__ == "__main__":
    example = "Patient [PATIENT], phone 9876543210."
    results = check_regex_leakage(example)
    print("Possible remaining structured PHI:", results)
