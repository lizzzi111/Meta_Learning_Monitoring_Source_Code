import evaluate
import nltk
import numpy as np

nltk.download('punkt')

def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds  = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def batch_tokenize_preprocess(batch, tokenizer, max_input_length, max_output_length):

    source = batch["input_sequence"]
    target = batch["output_sequence"]

    source_tokenized = tokenizer(
        source, padding="max_length",
        truncation=True, max_length=max_input_length
    )

    target_tokenized = tokenizer(
        target, padding="max_length",
        truncation=True, max_length=max_output_length
    )

    batch = {k: v for k, v in source_tokenized.items()}

    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in target_tokenized["input_ids"]
    ]

    return batch

def prepare_hg_ds(dataset, tokenizer, max_input_length, max_output_length, batch_size=8):
    """Tokenize and prepare the HF dataset."""
    return dataset.map(
        lambda batch: batch_tokenize_preprocess(
            batch,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        ),
        batch_size=batch_size,
        batched=True,
        remove_columns=dataset.column_names,
    )

def compute_metric_with_params(tokenizer, metrics_list=['rouge', 'bleu']):
    def compute_metrics(eval_preds):
    
        preds, labels = eval_preds
    
        if isinstance(preds, tuple):
            preds = preds[0]
    
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        # POST PROCESSING
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
        results_dict = {}
        for m in metrics_list:
            metric = evaluate.load(m)
    
            if m=='bleu':
                result = metric.compute(
                    predictions=decoded_preds, references=decoded_labels
                )
            elif m=='rouge':
                result = metric.compute(
                    predictions=decoded_preds, references=decoded_labels, use_stemmer=True
                )
            result = {key: value for key, value in result.items() if key!='precisions'}
    
            prediction_lens = [
                np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
            ]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            results_dict.update(result)
        return results_dict
    return compute_metrics