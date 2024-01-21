from __future__ import annotations  # noqa: EXE002, D100

import json
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    RobertaTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

from src.data_prep_utils import prep_for_hf
from src.model import load_model, load_tok
from src.processing_utils import compute_metric_with_params, prepare_hg_ds

if TYPE_CHECKING:
    from datasets import Dataset

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

EVAL_COLUMNS = [
    "eval_loss",
    "eval_rouge1",
    "eval_rouge2",
    "eval_rougeL",
    "eval_rougeLsum",
    "eval_bleu",
    "eval_gen_len",
]
LOGS_COLUMNS = [
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

def prepare_trainer(
        train_data: Dataset,
        validation_data: Dataset,
        training_arguments:dict,
        model: T5ForConditionalGeneration | str | int,
        tokenizer: RobertaTokenizerFast,
        ) -> list:
    """Prepare HuggingFace trainer."""
    analysis_name = None if "ANALYSIS_NAME" not in os.environ else os.environ["ANALYSIS_NAME"]
    model = load_model(model, freeze=True, analysis_name=analysis_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(device)

    training_args = Seq2SeqTrainingArguments(
        **training_arguments["SEQ_TRAINER_ARGS"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = compute_metric_with_params(tokenizer)

    return  Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
def set_analysis_name(training_args: dict) -> None:  # noqa: D103
    os.environ["ANALYSIS_NAME"] = f'{training_args["MODEL"]}_{training_args["TRAIN_N"]}_{training_args["BATCH_SIZE"]}'

def train(
        trainer:Seq2SeqTrainer,
        timestamp: int,
        save_model: bool=False,
        ) -> None:
    """Train the T5 Model."""
    # TRAINING
    analysis_name = os.environ["ANALYSIS_NAME"]
    trainer.train()
    if save_model:
        trainer.save_model(f"reports/{analysis_name}/trained_model")
    return trainer

def dd_eval(
        trainer: Seq2SeqTrainer,
        training_arguments: dict,
        ) -> dict:
    """Evaluate HF on validation data."""
    analysis_name = os.environ["ANALYSIS_NAME"]
    # ZERO - SHOT
    results = trainer.evaluate()
    results = pd.DataFrame(data=results, index=[0])[EVAL_COLUMNS]
    results.loc[0, :] = results.loc[0, :].apply(lambda x: round(x, 3))
    results.to_csv(f"reports/{analysis_name}/results.csv", index=False)
    logging.info(results)

    # STORE PARAMS
    with open(f"reports/{analysis_name}/config.json", "w") as params_file:  # noqa: PTH123
        json.dump(training_arguments, params_file)

    return {"loss" : results["eval_loss"], "rouge" : results["eval_rouge1"]}

def parse_logs(trainer: Seq2SeqTrainer) -> pd.DataFrame:
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
    return logs[LOGS_COLUMNS]

def nd_inference(
        df:pd.DataFrame,
        model: str,
        training_args: dict,
        ) -> pd.DataFrame:
    """Runs analysis on data drift along time dimension."""  # noqa: D401
    set_analysis_name(training_args)
    os.environ["ANALYSIS_NAME"] += "_nd_inference"
    analysis_name = os.environ["ANALYSIS_NAME"]
    tokenizer = load_tok(model, analysis_name)

    training_args["SEQ_TRAINER_ARGS"]["output_dir"] = f'reports/{os.environ["ANALYSIS_NAME"]}/results'
    training_args["SEQ_TRAINER_ARGS"]["logging_dir"] = f'reports/{os.environ["ANALYSIS_NAME"]}/logs'

    logging.info(f"Conduct {analysis_name} ANALYSIS")  # noqa: G004

    loss = []
    rouge_1 = []

    train_dataset = prep_for_hf(df, 0)
    train_data = prepare_hg_ds(
        train_dataset,
        tokenizer=tokenizer,
        max_input_length=training_args["ENCODER_LENGTH"],
        max_output_length=training_args["DECODER_LENGTH"],
    )

    for i in sorted(df.t_batch.unique()):

        test_dataset  = prep_for_hf(df, i+1)
        validation_data = prepare_hg_ds(
            test_dataset,
            tokenizer=tokenizer,
            max_input_length=training_args["ENCODER_LENGTH"],
            max_output_length=training_args["DECODER_LENGTH"],
        )

        trainer = prepare_trainer(
            train_data=train_data,
            validation_data=validation_data,
            training_arguments=training_args,
            model=model,
            tokenizer=tokenizer,
            )

        if i==0:
            logging.info(f"Training {i}")  # noqa: G004
            trainer = train(trainer=trainer, timestamp=i, save_model=True)

        logging.info(f"Validation for time step {i+1}")  # noqa: G004
        results = dd_eval(trainer=trainer, training_arguments=training_args)

        loss.append(results["loss"].iloc[0])
        rouge_1.append(results["rouge"].iloc[0])

    loss_df = pd.DataFrame(data={"i": range(df.t_batch.nunique()-1),
                    "loss" : loss,
                    "rouge": rouge_1})
    loss_df.to_csv(f"reports/{analysis_name}/ts_logs.csv", index=False)
    return loss_df

def retraining(
        df:pd.DataFrame,
        model: str,
        training_args: dict,
        ) -> pd.DataFrame:
    """Runs analysis on data drift along time dimension."""  # noqa: D401
    set_analysis_name(training_args)
    os.environ["ANALYSIS_NAME"] += "_retraining"
    analysis_name = os.environ["ANALYSIS_NAME"]
    tokenizer = load_tok(model, analysis_name)

    training_args["SEQ_TRAINER_ARGS"]["output_dir"] = f'reports/{os.environ["ANALYSIS_NAME"]}/results'
    training_args["SEQ_TRAINER_ARGS"]["logging_dir"] = f'reports/{os.environ["ANALYSIS_NAME"]}/logs'
    logging.info(f"Conduct {analysis_name} ANALYSIS")  # noqa: G004

    loss = []
    rouge_1 = []

    for i in range(1, df.t_batch.unique().max()):

        train_dataset = prep_for_hf(df, list(range(i)))
        train_data = prepare_hg_ds(
            train_dataset,
            tokenizer=tokenizer,
            max_input_length=training_args["ENCODER_LENGTH"],
            max_output_length=training_args["DECODER_LENGTH"],
        )

        test_dataset  = prep_for_hf(df, i)
        validation_data = prepare_hg_ds(
            test_dataset,
            tokenizer=tokenizer,
            max_input_length=training_args["ENCODER_LENGTH"],
            max_output_length=training_args["DECODER_LENGTH"],
        )

        trainer = prepare_trainer(
            train_data=train_data,
            validation_data=validation_data,
            training_arguments=training_args,
            model=model,
            tokenizer=tokenizer,
            )

        logging.info(f"Training {list(range(i))}")  # noqa: G004
        trainer = train(trainer=trainer, timestamp=i)

        logging.info(f"Validation for time step {i}")  # noqa: G004
        results = dd_eval(trainer=trainer, training_arguments=training_args)

        loss.append(results["loss"].iloc[0])
        rouge_1.append(results["rouge"].iloc[0])

    loss_df = pd.DataFrame(data={"i": range(df.t_batch.nunique()-1),
                    "loss" : loss,
                    "rouge": rouge_1})
    loss_df.to_csv(f"reports/{analysis_name}/ts_logs.csv", index=False)
    return loss_df

def continual(
        df:pd.DataFrame,
        model: str,
        training_args: dict,
        ) -> pd.DataFrame:
    """Runs analysis on data drift along time dimension."""  # noqa: D401
    set_analysis_name(training_args)
    os.environ["ANALYSIS_NAME"] += "_continual"
    analysis_name = os.environ["ANALYSIS_NAME"]
    tokenizer = load_tok(model, analysis_name)

    logging.info(f"Conduct {analysis_name} ANALYSIS")  # noqa: G004

    training_args["SEQ_TRAINER_ARGS"]["output_dir"] = f'reports/{os.environ["ANALYSIS_NAME"]}/results'
    training_args["SEQ_TRAINER_ARGS"]["logging_dir"] = f'reports/{os.environ["ANALYSIS_NAME"]}/logs'

    loss = []
    rouge_1 = []

    for i in range(1, df.t_batch.unique().max()):

        train_dataset = prep_for_hf(df, i-1)
        train_data = prepare_hg_ds(
            train_dataset,
            tokenizer=tokenizer,
            max_input_length=training_args["ENCODER_LENGTH"],
            max_output_length=training_args["DECODER_LENGTH"],
        )

        test_dataset  = prep_for_hf(df, i)
        validation_data = prepare_hg_ds(
            test_dataset,
            tokenizer=tokenizer,
            max_input_length=training_args["ENCODER_LENGTH"],
            max_output_length=training_args["DECODER_LENGTH"],
        )
        if i!=1:
            model=i
        trainer = prepare_trainer(
            train_data=train_data,
            validation_data=validation_data,
            training_arguments=training_args,
            model=model,
            tokenizer=tokenizer,
            )

        logging.info(f"Training {i-1}")  # noqa: G004
        trainer = train(trainer=trainer, timestamp=i, save_model=True)

        logging.info(f"Validation for time step {i}")  # noqa: G004
        results = dd_eval(trainer=trainer, training_arguments=training_args)

        loss.append(results["loss"].iloc[0])
        rouge_1.append(results["rouge"].iloc[0])

    loss_df = pd.DataFrame(data={"i": range(df.t_batch.nunique()-1),
                    "loss" : loss,
                    "rouge": rouge_1})
    loss_df.to_csv(f"reports/{analysis_name}/ts_logs.csv", index=False)
    return loss_df
