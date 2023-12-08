import pandas as pd
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

from src.processing_utils import compute_metric_with_params

EVAL_COLUMNS = [
                "eval_loss",
                "eval_rouge1",
                "eval_rouge2",
                "eval_rougeL",
                "eval_rougeLsum",
                "eval_bleu",
                "eval_gen_len",
                ]

def train_evaluate(train_data, validation_data, training_arguments, model, tokenizer):

    training_args = Seq2SeqTrainingArguments(
        **training_arguments
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = compute_metric_with_params(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ZERO - SHOT
    results_zero_shot = trainer.evaluate()
    results_zero_shot_df = pd.DataFrame(data=results_zero_shot, index=[0])[EVAL_COLUMNS]
    results_zero_shot_df.loc[0, :] = results_zero_shot_df.loc[0, :].apply(lambda x: round(x, 3))
    print(results_zero_shot_df)


    # TRAINING
    trainer.train()

    # FINE-TUNING
    results_fine_tune = trainer.evaluate()
    results_fine_tune_df = pd.DataFrame(data=results_fine_tune, index=[0])[EVAL_COLUMNS]
    results_fine_tune_df.loc[0, :] = results_fine_tune_df.loc[0, :].apply(lambda x: round(x, 3))
    print(results_fine_tune_df)