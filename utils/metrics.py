def compute_f1(prediction, ground_truth):
    """Return a dummy F1 score."""
    if prediction == ground_truth:
        return 1.0
    return 0.0