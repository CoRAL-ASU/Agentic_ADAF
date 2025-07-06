"""Utility functions for evaluating a model on a dataset."""

from utils.metrics import compute_f1  # type: ignore


def evaluate_on_dataset(model, dataset):
    """Return the average F1 score for the given dataset."""
    results = []
    for ex in dataset:
        pred = model.answer(ex["question"], ex["table"], ex.get("text"))
        score = compute_f1(pred, ex["answer"])
        results.append(score)
    return sum(results) / len(results)


if __name__ == "__main__":
    # TODO: load a trained model and dataset, then print evaluation metrics
    pass
