import numpy as np
import evaluate

sacrebleu_metric = evaluate.load("sacrebleu")

def compute_bleu_score(tokenizer, eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = sacrebleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return result