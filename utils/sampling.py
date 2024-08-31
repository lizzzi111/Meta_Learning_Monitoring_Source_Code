import pandas as pd
from datasets import load_dataset, Dataset
import utils.prep as pr
from sklearn.model_selection import KFold
from typing import Tuple
import numpy as np
import math

from transformers import AutoTokenizer

def create_splits(experiment_config : dict,
                  tokenizer: AutoTokenizer,
                  train_size: int=10000,
                  test_size: int=2500,
                  cluster_id: int=None) -> dict:

    DATA_STR = experiment_config["DATA_STR"]
    RS = experiment_config["RS"]
    DRIFT_TYPE = experiment_config["DRIFT_TYPE"]

    if DRIFT_TYPE=="no_drift":
        dataset = pd.read_csv(f"../data/processed/conala/{DATA_STR}/conala_mined_clustered.csv").drop_duplicates("question_id").reset_index(drop=True)
        dataset = dataset.sample(frac=1, random_state=RS).head(train_size+test_size).reset_index(drop=True)

        train_dataset = dataset.iloc[:train_size, :]
        test_dataset = dataset.iloc[train_size:, :]
        

    elif DRIFT_TYPE=="drift":
        dataset = pd.read_csv(f"../data/processed/conala/{DATA_STR}/conala_mined_clustered.csv").drop_duplicates("question_id").reset_index(drop=True)
        dataset = dataset.sample(frac=1, random_state=RS).head(train_size+test_size).reset_index(drop=True)

        train_dataset = dataset.iloc[:train_size, :]
        temp_df = dataset.loc[(dataset.cluster.isin([4, 1, 0]) & (-dataset.id.isin(train_dataset))), :].sample(frac=1, random_state=RS)

        if test_size <= temp_df.shape[0]: 
            test_dataset = temp_df[:test_size, :].reset_index(drop=True)
        else: 
            missing_n = test_size - temp_df.shape[0]
            test_dataset = temp_df.copy()
            major_df = dataset.loc[(-dataset.id.isin(test_dataset) & (-dataset.id.isin(train_dataset))), :].sample(n=missing_n, random_state=RS)
            test_dataset = pd.concat([test_dataset, major_df]).sample(frac=1, random_state=RS).reset_index(drop=True)
            

    print("Train Data: ", train_dataset.shape)
    print("Test Data: ", test_dataset.shape)

    print("Train Data: Cluster", train_dataset.cluster.value_counts())
    print("Test Data: Cluster", test_dataset.cluster.value_counts())

    train_dataset = Dataset.from_pandas(train_dataset.sample(frac=1, random_state=RS).reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_dataset.sample(frac=1, random_state=RS).reset_index(drop=True))

    train_dataset = pr.preprocess_dataset(train_dataset, tokenizer=tokenizer, intent_colum_name="intent")
    test_data = pr.preprocess_dataset(test_dataset, tokenizer=tokenizer, intent_colum_name="intent")
    
    test_df = pd.DataFrame(test_data)
    test_df["id"] = test_df.index

    train_df = pd.DataFrame(train_dataset)
    train_df["id"] = train_df.index

    return {"train_data": train_dataset,
            "test_data": test_data,
            "test_df": test_df,
            "train_df" : train_df}

def prep_cv_validation(train_dataset: Dataset,
                       experiment_config : dict) -> Tuple:
    
    NFOLD = experiment_config["NFOLD"]
    RS = experiment_config["RS"]

    # Cross Validation
    folds = KFold(n_splits=NFOLD, random_state=RS, shuffle=True)
    questions_list = np.array(list(set(train_dataset["question_id"])))
    splits_obj = folds.split(questions_list)
    splits = []
    for i, (train_idxs, val_idxs) in enumerate(splits_obj):
        print(f"Fold {i}")
        splits.append([train_idxs, val_idxs])
    
    return splits, questions_list