import re


# Structured patterns are supplementary detectors, not a replacement
# for trained NER models. Validate patterns against your dataset.
REGEX_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "PHONE": r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}(?!\w)",
    "DATE": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    "ID": r"\b(?:MRN|Medical Record Number|Patient ID)\s*[:#-]?\s*[A-Za-z0-9-]+\b"
}


def regex_predictions(text):
    spans = []
    for label, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), label, 1.0))
    return spans


def merge_predictions(text, model_spans, regex_spans=None):
    """
    Inputs use (start, end, label, confidence).
    Regex spans take priority over overlapping model spans.
    """
    candidates = [(s, e, lab, float(conf), False)
                  for s, e, lab, conf in model_spans]
    if regex_spans is None:
        regex_spans = regex_predictions(text)
    candidates.extend(
        (s, e, lab, float(conf), True)
        for s, e, lab, conf in regex_spans
    )

    candidates.sort(
        key=lambda x: (x[4], x[3], x[1] - x[0]),
        reverse=True
    )

    selected = []
    for start, end, label, confidence, is_regex in candidates:
        if start < 0 or end > len(text) or start >= end:
            continue
        overlaps = any(
            not (end <= old_start or start >= old_end)
            for old_start, old_end, _, _ in selected
        )
        if not overlaps:
            selected.append((start, end, label, confidence))

    return sorted(selected, key=lambda x: (x[0], x[1]))


def redact_text(text, spans):
    output = text
    for start, end, label, confidence in sorted(spans, key=lambda x: x[0], reverse=True):
        output = output[:start] + "[{}]".format(label) + output[end:]
    return output


if __name__ == "__main__":
    sample = "Patient contact: 9876543210 and john@example.com on 12/05/2025."
    regex_spans = regex_predictions(sample)
    print("Detected:", regex_spans)
    print("Redacted:", redact_text(sample, regex_spans))
