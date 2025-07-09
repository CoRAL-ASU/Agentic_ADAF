"""Utility metric functions."""

from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Split a string into whitespace-separated tokens."""

    return text.split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Return token-level F1 for two strings.

    Parameters
    ----------
    prediction:
        Model predicted answer.
    ground_truth:
        Reference answer.

    Returns
    -------
    float
        Harmonic mean of precision and recall computed on tokens.
    """

    pred_tokens = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)
