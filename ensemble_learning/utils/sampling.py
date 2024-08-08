import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
import torch
import evaluate
import utils.prep as pr
from sklearn.model_selection import KFold
from typing import Tuple
import numpy as np

from transformers import AutoTokenizer

def create_splits(experiment_config : dict,
                  tokenizer: AutoTokenizer,
                  test: bool=False) -> Tuple:

    DATE_STR = experiment_config["DATE_STR"]
    RS = experiment_config["RS"]
    DRIFT_TYPE = experiment_config["DRIFT_TYPE"]

    if DRIFT_TYPE=="no_drift":
        dataset = pd.read_csv(f"../data/processed/conala/{DATE_STR}/conala_mined_clustered.csv").head(10000)
        qids = sorted(dataset.question_id.unique())
        train_idx, test_idx = qids[:int(len(qids)*0.785)], qids[int(len(qids)*0.785):]
        test_idx.append(train_idx[-1])
        train_idx.pop(-1)
        train_dataset = dataset[dataset.question_id.isin(train_idx)]
        test_dataset = dataset[dataset.question_id.isin(test_idx)]
        

    elif DRIFT_TYPE=="sudden":
        dataset = pd.read_csv(f"../data/processed/conala/{DATE_STR}/conala_mined_clustered.csv")
        dataset_4_cl = dataset[dataset.cluster==4].sample(n=2000, random_state=RS)
        dataset_non_4_cl = dataset[dataset.cluster!=4].sample(n=8000, random_state=RS)

        qids_4_cl = sorted(dataset_4_cl.question_id.unique())
        train_idx_4_cl, test_idx_4_cl = qids_4_cl[int(len(qids_4_cl)*0.99):], qids_4_cl[:int(len(qids_4_cl)*0.99)]

        qids_non4_cl = sorted(dataset_non_4_cl.question_id.unique())
        train_idx_non4_cl, test_idx_non4_cl = qids_non4_cl[:int(len(qids_non4_cl)*0.99)], qids_non4_cl[int(len(qids_non4_cl)*0.99):]

        train_dataset_4cl = dataset_4_cl[dataset_4_cl.question_id.isin(train_idx_4_cl)]
        test_dataset_4cl = dataset_4_cl[dataset_4_cl.question_id.isin(test_idx_4_cl)]

        train_dataset_non4cl = dataset_non_4_cl[dataset_non_4_cl.question_id.isin(train_idx_non4_cl)]
        test_dataset_non4cl = dataset_non_4_cl[dataset_non_4_cl.question_id.isin(test_idx_non4_cl)]

        train_dataset = pd.concat([train_dataset_4cl, train_dataset_non4cl], axis=0).sample(frac=1, random_state=RS).reset_index(drop=True)
        test_dataset = pd.concat([test_dataset_4cl, test_dataset_non4cl], axis=0).sample(frac=1, random_state=RS).reset_index(drop=True)

    elif DRIFT_TYPE=="slight":
        dataset = pd.read_csv(f"../data/processed/conala/{DATE_STR}/conala_mined_clustered.csv")
        dataset_4_cl = dataset[dataset.cluster==4].sample(n=2000, random_state=RS)
        dataset_non_4_cl = dataset[dataset.cluster!=4].sample(n=8000, random_state=RS)

        qids_4_cl = sorted(dataset_4_cl.question_id.unique())
        train_idx_4_cl, test_idx_4_cl = qids_4_cl[int(len(qids_4_cl)*0.9):], qids_4_cl[:int(len(qids_4_cl)*0.9)]

        qids_non4_cl = sorted(dataset_non_4_cl.question_id.unique())
        train_idx_non4_cl, test_idx_non4_cl = qids_non4_cl[:int(len(qids_non4_cl)*0.968)], qids_non4_cl[int(len(qids_non4_cl)*0.968):]

        test_idx_non4_cl.append(train_idx_non4_cl[0])
        train_idx_non4_cl.pop(0)
        
        train_dataset_4cl = dataset_4_cl[dataset_4_cl.question_id.isin(train_idx_4_cl)]
        test_dataset_4cl = dataset_4_cl[dataset_4_cl.question_id.isin(test_idx_4_cl)]

        train_dataset_non4cl = dataset_non_4_cl[dataset_non_4_cl.question_id.isin(train_idx_non4_cl)]
        test_dataset_non4cl = dataset_non_4_cl[dataset_non_4_cl.question_id.isin(test_idx_non4_cl)]

        train_dataset = pd.concat([train_dataset_4cl, train_dataset_non4cl], axis=0).sample(frac=1, random_state=RS).reset_index(drop=True)
        test_dataset = pd.concat([test_dataset_4cl, test_dataset_non4cl], axis=0).sample(frac=1, random_state=RS).reset_index(drop=True)

    print("Train Data: ", train_dataset.shape)
    print("Test Data: ", test_dataset.shape)

    print("Train Data: Cluster", train_dataset.cluster.value_counts())
    print("Test Data: Cluster", test_dataset.cluster.value_counts())

    train_dataset = Dataset.from_pandas(train_dataset.sample(frac=1, random_state=RS).reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_dataset.sample(frac=1, random_state=RS).reset_index(drop=True))

    if test:
        train_dataset = pr.preprocess_dataset(train_dataset, tokenizer=tokenizer, intent_colum_name="intent")
    test_data = pr.preprocess_dataset(test_dataset, tokenizer=tokenizer, intent_colum_name="intent")
    
    test_df = pd.DataFrame(test_data)
    test_df["id"] = test_df.index

    return {"train_data": train_dataset,
            "test_data": test_data,
            "test_df": test_df}

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