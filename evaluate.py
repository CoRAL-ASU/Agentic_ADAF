def evaluate_on_dataset(model, dataset):
    from utils.metrics import compute_f1
    results = []
    for ex in dataset:
        pred = model.answer(ex['question'], ex['table'], ex.get('text'))
        score = compute_f1(pred, ex['answer'])
        results.append(score)
    return sum(results)/len(results)