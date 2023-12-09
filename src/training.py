import json

import pandas as pd
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.processing_utils import compute_metric_with_params, prepare_hg_ds
from src.data_prep_utils import prep_for_hf

EVAL_COLUMNS = [
    "eval_loss",
    "eval_rouge1",
    "eval_rouge2",
    "eval_rougeL",
    "eval_rougeLsum",
    "eval_bleu",
    "eval_gen_len",
]

def prepare_trainer(train_data, validation_data, training_arguments, model, tokenizer):

    analysis_name = f'{training_arguments["MODEL"]}_{training_arguments["TRAIN_N"]}_{training_arguments["BATCH_SIZE"]}'
    training_args = Seq2SeqTrainingArguments(
        **training_arguments["SEQ_TRAINER_ARGS"],
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
    return trainer, analysis_name


def eval(trainer, train_data, validation_data, training_arguments, model, tokenizer):
    #trainer, analysis_name = prepare_trainer(train_data, validation_data, training_arguments, model, tokenizer)
    analysis_name = f'{training_arguments["MODEL"]}_{training_arguments["TRAIN_N"]}_{training_arguments["BATCH_SIZE"]}'
    # ZERO - SHOT
    results_zero_shot = trainer.evaluate()
    results_zero_shot_df = pd.DataFrame(data=results_zero_shot, index=[0])[EVAL_COLUMNS]
    results_zero_shot_df.loc[0, :] = results_zero_shot_df.loc[0, :].apply(lambda x: round(x, 3))
    results_zero_shot_df.to_csv(f"reports/{analysis_name}/zero_shot_results.csv", index=False)
    print(results_zero_shot_df)

    # STORE PARAMS
    with open(f"reports/{analysis_name}/config.json", "w") as params_file:
        json.dump(training_arguments, params_file)

    return trainer, {"loss" : results_zero_shot_df["eval_loss"], "rouge" : results_zero_shot_df["eval_rouge1"]}, analysis_name


def train_evaluate(train_data, validation_data, training_arguments, model, tokenizer):
    trainer, analysis_name = eval(train_data, validation_data, training_arguments, model, tokenizer)

    # TRAINING
    trainer.train()

    # FINE-TUNING
    results_fine_tune = trainer.evaluate()
    results_fine_tune_df = pd.DataFrame(data=results_fine_tune, index=[0])[EVAL_COLUMNS]
    results_fine_tune_df.loc[0, :] = results_fine_tune_df.loc[0, :].apply(lambda x: round(x, 3))
    results_fine_tune_df.to_csv(f"reports/{analysis_name}/fine_tune_results.csv", index=False)
    print(results_fine_tune_df)

    logs_df = parse_logs(trainer)
    logs_df.to_csv(f"reports/{analysis_name}/training_logs.csv", index=False)

    return trainer

def train(train_data, validation_data, training_arguments, model, tokenizer):
    trainer, _ = prepare_trainer(train_data, validation_data, training_arguments, model, tokenizer)
    # TRAINING
    trainer.train()

    return trainer


def parse_logs(trainer):
    """Parse Trainer Logs."""
    log_history = trainer.state.log_history
    train_log = pd.DataFrame(columns=log_history[0].keys())
    eval_log = pd.DataFrame(columns=log_history[1].keys())

    for log in log_history:
        if "loss" in log:
            train_log = pd.concat(
                [train_log, pd.DataFrame.from_dict(log, orient="index").T],
                axis=0,
            )
        elif "eval_loss" in log:
            eval_log = pd.concat(
                [eval_log, pd.DataFrame.from_dict(log, orient="index").T],
                axis=0,
            )

    logs = train_log.merge(
        eval_log,
        how="inner",
        left_on=["epoch", "step"],
        right_on=["epoch", "step"],
    )
    return logs[
        [
            "epoch",
            "loss",
            "step",
            "eval_loss",
            "eval_rouge1",
            "eval_rouge2",
            "eval_rougeL",
            "eval_rougeLsum",
            "eval_bleu",
        ]
    ]

def dd_analysis(df, tokenizer, model, training_args):
    loss = []
    rouge_1 = []

    train_dataset = prep_for_hf(df, -1)
    train_data = prepare_hg_ds(
        train_dataset,
        tokenizer=tokenizer,
        max_input_length=training_args["ENCODER_LENGTH"],
        max_output_length=training_args["DECODER_LENGTH"],
    )

    for i in range(df.t_batch.nunique()-1):
        print(f"Validation for time step {i}")
        test_dataset  = prep_for_hf(df, 0)
        validation_data = prepare_hg_ds(
            test_dataset,
            tokenizer=tokenizer,
            max_input_length=training_args["ENCODER_LENGTH"],
            max_output_length=training_args["DECODER_LENGTH"],
        )
        if i==0:
            trainer = train(train_data, validation_data, training_args, model, tokenizer)
        trainer, results, analysis_name = eval(trainer, train_data, validation_data, training_args, model, tokenizer)
        loss.append(results["loss"])
        rouge_1.append(results["rouge"])
    loss_df = pd.DataFrame(data={"i": range(df.t_batch.nunique()-1),
                    "loss" : loss,
                    "rouge": rouge_1})
    loss_df.to_csv(f"reports/{analysis_name}/ts_logs.csv", index=False)
    return loss_df
