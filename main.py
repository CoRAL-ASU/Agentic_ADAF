from agents.coordinator import AdaptiveOrchestrator
from utils.data_loader import load_benchmark
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tatqa')
    parser.add_argument('--llm', type=str, default='mistral-7b')
    args = parser.parse_args()

    data = load_benchmark(args.dataset)
    orchestrator = AdaptiveOrchestrator(model_name=args.llm)
    for sample in data:
        result = orchestrator.run(sample)
        print(f"Q: {sample['question']}\nA: {result['answer']}\n")