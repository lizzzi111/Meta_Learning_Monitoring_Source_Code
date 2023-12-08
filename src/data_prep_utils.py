import math
import os
import logging

import pandas as pd
from datasets import load_from_disk, Dataset

logger = logging.getLogger()

def load_time_sorted_conala(path: str) -> pd.DataFrame:
    
    """
    Load local conala data set, convert it to pandas and return sorted df along the time axis (here question id).
    """
    dataset_curated = load_from_disk(path)
    train_df = pd.DataFrame(data={'question_id': dataset_curated['train']['question_id'], 
                                'intent' : dataset_curated['train']['intent'],
                                'rewritten_intent' : dataset_curated['train']['rewritten_intent'],
                                'snippet' : dataset_curated['train']['snippet']})

    test_df = pd.DataFrame(data={'question_id': dataset_curated['test']['question_id'], 
                                'intent' : dataset_curated['test']['intent'],
                                'rewritten_intent' : dataset_curated['test']['rewritten_intent'],
                                'snippet' : dataset_curated['test']['snippet']})

    full_df = pd.concat([train_df, test_df], axis=0)
    full_df = full_df.sort_values("question_id").reset_index(drop=True)
    print(f"Unique questions: {full_df.question_id.nunique()}")
    return full_df

def conala_to_time_batches(full_df:pd.DataFrame, train_size: int, batch_size: int) -> pd.DataFrame:

    """
    Prepare to time line batching of conala dataset
    """

    print("Sort Question IDs")
    qids = full_df.question_id.unique()
    qids.sort()
    
    print("Create t=0 training sample")

    first_train_ids = qids[:train_size]
    batches = []
    batch_n = math.ceil((full_df.question_id.nunique()-train_size)/batch_size)

    for i in range(batch_n):
        #print(i)

        batch_start = train_size+(i)*batch_size 
        #print(batch_start)
        
        if i!=(batch_n-1):
            batch_end = batch_start + batch_size
            batches.append(qids[batch_start:batch_end])
            #print(batch_end)
        else: 
            batches.append(qids[batch_start:])
            #print(len(qids)-1)
    
    full_df['t_batch'] = -1

    for i, batch_ids in enumerate(batches):
        full_df.loc[full_df.question_id.isin(set(batch_ids)), 't_batch'] = i

    return full_df

def prep_for_hf(df: pd.DataFrame, batch_id: int): 
    """
    Convert pandas to huggingface
    """
    df = df.rename(columns={'snippet': 'input_sequence', 
                    'intent' : 'output_sequence'})
    df = df.loc[df.t_batch==batch_id, ['input_sequence', 'output_sequence']]
    df = df.sample(frac=1, random_state=42)
    return Dataset.from_pandas(df)