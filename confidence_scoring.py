
def calculate_confidence(predictions):
    """
    Calculate the average confidence of PHI predictions.

    Each prediction must have this format:
    (start, end, label, confidence)
    """

    if not predictions:
        return 0.0

    scores = []

    for prediction in predictions:
        if len(prediction) >= 4:
            score = float(prediction[3])
            score = max(0.0, min(1.0, score))
            scores.append(score)

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 4)


def confidence_level(score):
    """Convert a confidence score into a readable category."""

    if score >= 0.90:
        return "High"
    elif score >= 0.70:
        return "Medium"
    else:
        return "Low"
