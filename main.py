"""Entry point for running the adaptive table QA pipeline."""

import argparse

# These imports expect the full project layout from the README.  They are left
# as-is because the repository only contains partial code.
from agents.coordinator import AdaptiveOrchestrator  # type: ignore
from utils.data_loader import load_benchmark  # type: ignore


def main() -> None:
    """Run the orchestrator over a benchmark dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tatqa")
    parser.add_argument("--llm", type=str, default="mistral-7b")
    args = parser.parse_args()

    data = load_benchmark(args.dataset)
    orchestrator = AdaptiveOrchestrator(model_name=args.llm)
    for sample in data:
        result = orchestrator.run(sample)
        print(f"Q: {sample['question']}\nA: {result['answer']}\n")


if __name__ == "__main__":
    main()
