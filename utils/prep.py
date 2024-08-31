import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

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

def preprocess_dataset(dataset: Dataset,
               tokenizer: AutoTokenizer,
               max_input_length:int=15, 
               max_output_length:int=20,
               intent_colum_name:str="rewritten_intent") -> Dataset:

    dataset = dataset.rename_column("snippet", "input_sequence")
    dataset = dataset.rename_column(intent_colum_name, "output_sequence")

    dataset = dataset.filter(lambda x: pd.notna(x["input_sequence"]))
    dataset = dataset.filter(lambda x: pd.notna(x["output_sequence"]))

    dataset = dataset.map(
        lambda batch: batch_tokenize_preprocess(
            batch,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        ),
        batch_size=4,
        batched=True,
    )
    return dataset