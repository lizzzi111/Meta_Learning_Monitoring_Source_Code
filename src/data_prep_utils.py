#!/usr/bin/python3
"""Helper functions to prepare the data for analysis."""

from __future__ import annotations

import logging
import math

import pandas as pd
from datasets import Dataset, load_from_disk

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def load_time_sorted_conala(path: str) -> pd.DataFrame:
    """Load local conala data set, convert it to pandas and return sorted df along the time axis (here question id)."""
    dataset_curated = load_from_disk(path)
    train_df = pd.DataFrame(data={"question_id": dataset_curated["train"]["question_id"],
                                "intent" : dataset_curated["train"]["intent"],
                                "rewritten_intent" : dataset_curated["train"]["rewritten_intent"],
                                "snippet" : dataset_curated["train"]["snippet"]})

    test_df = pd.DataFrame(data={"question_id": dataset_curated["test"]["question_id"],
                                "intent" : dataset_curated["test"]["intent"],
                                "rewritten_intent" : dataset_curated["test"]["rewritten_intent"],
                                "snippet" : dataset_curated["test"]["snippet"]})

    full_df = pd.concat([train_df, test_df], axis=0)
    full_df = full_df.sort_values("question_id").reset_index(drop=True)
    logging.info(f"Unique questions: {full_df.question_id.nunique()}")  # noqa: G004
    return full_df

def conala_to_time_batches(full_df:pd.DataFrame, 
                           train_size: int, 
                           n_batches: int=None,
                           batch_size: int=None) -> pd.DataFrame:
    """Prepare to time line batching of conala dataset."""
    logging.info("Sort Question IDs")
    qids = full_df.question_id.unique()
    qids.sort()

    logging.info("Create t=0 training sample")

    if n_batches and batch_size:
        raise ValueError("Please provide either n_batches or batch_size, not both.")
    if not n_batches and not batch_size:
        raise ValueError("Please provide either n_batches or batch_size.")
    
    batches = []

    if n_batches:
        batch_size = math.ceil((full_df.question_id.nunique()-train_size)/n_batches)
    #first_train_ids = qids[:train_size]
    else:
        n_batches = math.ceil((full_df.question_id.nunique()-train_size)/batch_size)

    for i in range(n_batches):
        #print(i)

        batch_start = train_size+(i)*batch_size
        #print(batch_start)

        if i!=(n_batches-1):
            batch_end = batch_start + batch_size
            batches.append(qids[batch_start:batch_end])
            #print(batch_end)
        else:
            batches.append(qids[batch_start:])
            #print(len(qids)-1)

    full_df["t_batch"] = 0

    for i, batch_ids in enumerate(batches):
        full_df.loc[full_df.question_id.isin(set(batch_ids)), "t_batch"] = i+1

    return full_df

def prep_for_hf(df: pd.DataFrame, batch_id: int|list) -> Dataset:
    """Convert pandas dataframe to huggingface."""
    df = df.rename(columns={"snippet": "input_sequence",  # noqa: PD901
                    "intent" : "output_sequence"})
    if isinstance(batch_id, list):
        df = df.loc[df.t_batch.isin(batch_id), ["input_sequence", "output_sequence"]]  # noqa: PD901
    elif isinstance(batch_id, int):
        df = df.loc[df.t_batch==batch_id, ["input_sequence", "output_sequence"]]  # noqa: PD901
    df = df.sample(frac=1, random_state=42)  # noqa: PD901
    return Dataset.from_pandas(df)
