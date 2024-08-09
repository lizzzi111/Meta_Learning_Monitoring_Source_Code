
import pandas as pd
import numpy as np
import pickle 
from datasets import DatasetDict, Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import copy
import os
import math
import torch
import evaluate
from typing import Tuple

import utils.prep as pr
import utils.eval as ev
import utils.inference as infer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.sparse import hstack

def generate_summaries_batches(text: list, 
                               model: AutoModelForSeq2SeqLM, 
                               tokenizer: AutoTokenizer, 
                               TRAIN_ARGS: dict) -> list:
    summaries = []
            
    if len(text)>1000:
        
        batch_size = 1000
        n_batches = math.ceil(len(text)/batch_size)

        for batch in range(n_batches):

            batch_start_idx = batch*batch_size
            batch_end_idx = batch*batch_size + batch_size

            if batch==(n_batches-1):
                batch_end_idx = len(text)
            summary = infer.generate_summary(text[batch_start_idx:batch_end_idx],
                                            model,
                                            tokenizer,
                                            TRAIN_ARGS["ENCODER_LENGTH"],
                                            TRAIN_ARGS["DECODER_LENGTH"])[1]
            summaries.append(summary)

        summaries = [sentence for summary_list in summaries for sentence in summary_list]
        
        prediction = summaries
    else: 
        summaries = infer.generate_summary(text, 
                                        model,
                                        tokenizer,
                                        TRAIN_ARGS["ENCODER_LENGTH"],
                                        TRAIN_ARGS["DECODER_LENGTH"])
        prediction = summaries[1]

    return prediction

def cv_training_epochs_sets(experiment_config:dict,
                            splits: list,
                            questions_list: list,
                            train_dataset: Dataset,
                            tokenizer: AutoTokenizer) -> pd.DataFrame:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rouge = evaluate.load('rouge')

    FULL_TRAIN_ARGS = experiment_config["FULL_TRAIN_ARGS"]
    MODEL_NAME = experiment_config["MODEL_NAME"]

    #### PREPARE THE RESULTS DICTIONARY
    fold_results = {}
    for epoch_i, epoch_set in enumerate(sorted(FULL_TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"])):
        fold_results[epoch_set] = {}

    #### VALIDATION LOOP
    for i, (train_idxs, val_idxs) in enumerate(splits):

        print(f"Fold {i}")
        fold_dataset = DatasetDict({
            "train": train_dataset.filter(lambda q_id: q_id["question_id"] in questions_list[train_idxs]),
            "validation": train_dataset.filter(lambda q_id: q_id["question_id"] in questions_list[val_idxs]),
        })
        fold_train = pr.preprocess_dataset(fold_dataset["train"], tokenizer=tokenizer, intent_colum_name="intent")
        fold_val = pr.preprocess_dataset(fold_dataset["validation"], tokenizer=tokenizer, intent_colum_name="intent")
        

        for epoch_i, epoch_set in enumerate(sorted(FULL_TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"])):

            fold_df = pd.DataFrame(fold_val)
            print(f"TRAINING EPOCH SET {epoch_set}")

            TRAIN_ARGS = copy.deepcopy(FULL_TRAIN_ARGS)
            FOLD_MODEL_PATH = "./tmp/"

            if epoch_set > 1: 
                TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"] = epoch_set - latest_run_epoch
            else:
                TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"] = epoch_set
            
            print(f'TRAINING EPOCHS {TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"]}')

            if epoch_set > 1: 
                model = AutoModelForSeq2SeqLM.from_pretrained(FOLD_MODEL_PATH)
                print(f"LOADING MODEL {FOLD_MODEL_PATH}")
            else: 
                model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
                print(f"LOADING MODEL {MODEL_NAME}")

            print(device)
            model.to(device)

            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
            compute_metrics = ev.compute_metric_with_params(tokenizer) 

            if not os.path.exists(f'reports/'): 
                os.mkdir(f'reports/')

            training_args = Seq2SeqTrainingArguments(
                    **TRAIN_ARGS["SEQ_TRAINER_ARGS"],
                )
            
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=fold_train,
                eval_dataset=fold_val,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            if epoch_set!=0:
                trainer.train()

            text = fold_val["input_sequence"]
            fold_df["prediction"] = generate_summaries_batches(text=text,
                                                                model=model, 
                                                                tokenizer=tokenizer,
                                                                TRAIN_ARGS=TRAIN_ARGS)
            


            fold_df["rouge"] = rouge.compute(predictions=fold_df["prediction"], 
                                references=fold_df["output_sequence"],
                                use_stemmer=True, 
                                use_aggregator=False,
                                rouge_types=["rouge1"])["rouge1"]
            
            fold_results[epoch_set][i] = fold_df
            
            ########## SAVE FOLD MODEL
            if not os.path.exists(FOLD_MODEL_PATH): 
                os.mkdir(FOLD_MODEL_PATH)

            trainer.save_model(FOLD_MODEL_PATH)
            latest_run_epoch = epoch_set

    return fold_results


def cv_cluster_set(experiment_config:dict,
                    splits: list,
                    questions_list: list,
                    train_dataset: Dataset,
                    tokenizer: AutoTokenizer,
                    fold_results: dict,
                    cluster_id: int,) -> pd.DataFrame:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rouge = evaluate.load('rouge')

    FULL_TRAIN_ARGS = experiment_config["FULL_TRAIN_ARGS"]
    MODEL_NAME = experiment_config["MODEL_NAME"]
    CLUSTER_EPOCHS = experiment_config["CLUSTER_EPOCHS"]
    DATE_STR = experiment_config["DATE_STR"]
    RS = experiment_config["RS"]

    #### LOAD CLUSTER_ID 
    # [5291, 5295, 5298]
    dataset = pd.read_csv(f"../data/processed/conala/{DATE_STR}/conala_mined_clustered.csv")
    cluster_fold_train = dataset[dataset.cluster==cluster_id] 
    
    #### PREPARE THE RESULTS DICTIONARY
    fold_results[f"cluster_{cluster_id}"] = {}

    #### VALIDATION LOOP
    for i, (train_idxs, val_idxs) in enumerate(splits):

        print(f"Fold {i}")
        fold_dataset = DatasetDict({
            "train": train_dataset.filter(lambda q_id: q_id["question_id"] in questions_list[train_idxs]),
            "validation": train_dataset.filter(lambda q_id: q_id["question_id"] in questions_list[val_idxs]),
        })
        
        fold_train = cluster_fold_train.loc[~cluster_fold_train.question_id.isin(train_dataset["question_id"]),:] 
        fold_train = Dataset.from_pandas(fold_train.sample(n=5291, random_state=RS).reset_index(drop=True))
        fold_train = pr.preprocess_dataset(fold_train, tokenizer=tokenizer, intent_colum_name="intent")

        fold_val = pr.preprocess_dataset(fold_dataset["validation"], tokenizer=tokenizer, intent_colum_name="intent")
        
        fold_df = pd.DataFrame(fold_val)
        print(f"TRAINING CLUSTER SET {cluster_id} FOR EPOCHS{CLUSTER_EPOCHS}")

        TRAIN_ARGS = copy.deepcopy(FULL_TRAIN_ARGS)
        FOLD_MODEL_PATH = "./tmp/"

        TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"] = CLUSTER_EPOCHS
            
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print(f"LOADING MODEL {MODEL_NAME}")

        print(device)
        model.to(device)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        compute_metrics = ev.compute_metric_with_params(tokenizer) 

        if not os.path.exists(f'reports/'): 
            os.mkdir(f'reports/')

        training_args = Seq2SeqTrainingArguments(
                **TRAIN_ARGS["SEQ_TRAINER_ARGS"],
            )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=fold_train,
            eval_dataset=fold_val,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        if i==0:
            trainer.train()

            ########## SAVE CLUSTER MODEL
            if not os.path.exists(FOLD_MODEL_PATH): 
                os.mkdir(FOLD_MODEL_PATH)

            trainer.save_model(FOLD_MODEL_PATH)

        text = fold_val["input_sequence"]
        fold_df["prediction"] = generate_summaries_batches(text=text,
                                                            model=model, 
                                                            tokenizer=tokenizer,
                                                            TRAIN_ARGS=TRAIN_ARGS)
        


        fold_df["rouge"] = rouge.compute(predictions=fold_df["prediction"], 
                            references=fold_df["output_sequence"],
                            use_stemmer=True, 
                            use_aggregator=False,
                            rouge_types=["rouge1"])["rouge1"]
            
        fold_results[f"cluster_{cluster_id}"][i] = fold_df
            
    return fold_results


def results_dict_todf(fold_results:dict) -> pd.DataFrame:
    for epoch_i, (epoch_set) in enumerate(fold_results.keys()): 
    
        for i, (k, f_df) in enumerate(fold_results[epoch_set].items()): 
            
            f_df['fold'] = k
            f_df['model_set'] = epoch_set

            if (epoch_i==0 and i==0): 
                cv_df = f_df.copy()
            else: 
                cv_df = pd.concat([cv_df, f_df])

    return cv_df


def step_two(experiment_config, 
             X_train,
             y_train,
             model,
             X_val=None,
             y_val=None,
             save=False): 
    
    ANALYSIS_POSTFIX = experiment_config["ANALYSIS_POSTFIX"]
    if model=="lr":
        reg = LinearRegression().fit(X_train, y_train)
    elif model =="svm": 
        reg = SVR().fit(X_train, y_train)
    elif model=="rf":
        reg = RandomForestRegressor.fit(X_train, y_train)
    elif model=="lgbm":
        reg = LGBMRegressor()
        reg.fit(X=X_train, y=y_train)
    elif model=="catboost":
        reg = CatBoostRegressor()
        reg.fit(X=X_train, y=y_train)

    if save:
        with open(f'./models/reg_{model}_{ANALYSIS_POSTFIX}.pkl','wb') as f:
            pickle.dump(reg, f)
        return f'./models/reg_{model}_{ANALYSIS_POSTFIX}.pkl'
    
    else:
        y_pred = reg.predict(X_val)
        y_pred[y_pred<0] = 0
        mae = mean_absolute_error(y_true=y_val, y_pred=y_pred)
        rmse = math.sqrt(mean_squared_error(y_true=y_val, y_pred=y_pred))
        return {"pred": y_pred, "mae": mae, "rmse": rmse}
    

def cv_step_2(experiment_config:dict, cv_df:pd.DataFrame) -> Tuple:

    t_models = ["lr", "svm", "lgbm", "catboost"]

    results = {}


    for test_fold in range(cv_df.fold.max()+1):
        print(test_fold)

        # Prepare the input data
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(cv_df.loc[cv_df.fold!=test_fold, "input_sequence"])
        X_train_column_sparse = pd.get_dummies(cv_df.loc[cv_df.fold!=test_fold, "model_set"], sparse=True).sparse.to_coo().tocsr()
        X_train = hstack([X_train_column_sparse, X_train_tfidf])
        y_train = cv_df.loc[cv_df.fold!=test_fold, "rouge"]
        
        X_val_tfidf = vectorizer.transform(cv_df.loc[cv_df.fold==test_fold, "input_sequence"])
        X_val_column_sparse = pd.get_dummies(cv_df.loc[cv_df.fold==test_fold, "model_set"], sparse=True).sparse.to_coo().tocsr()
        X_val = hstack([X_val_column_sparse, X_val_tfidf])
        y_val = cv_df.loc[cv_df.fold==test_fold, "rouge"]

        results[test_fold] = {}
        for model in t_models:
            print(model)
            preds_df = step_two(experiment_config=experiment_config,
                                X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val,
                                model=model)
            cv_df.loc[cv_df.fold==test_fold, f"{model}_perf_hat"] = preds_df["pred"]
            results[test_fold][model] = preds_df

    cv_df = cv_df.reset_index(drop=True)

    # ENSEMBLE ESTIMATE (JUST HIGHEST PREDICTIONS)
    models_index = cv_df.groupby("id")["catboost_perf_hat"].idxmax()
    optimal_ensemble = cv_df.iloc[models_index][["id", "model_set"]]
    optimal_ensemble_map = dict(zip(optimal_ensemble.id, optimal_ensemble.model_set))
    cv_df["opt_es_id"] = cv_df.id.map(optimal_ensemble_map)
    ensemble_preds = cv_df.loc[cv_df["model_set"]==cv_df["opt_es_id"], :]
    ensemble_preds["rouge"].mean()
    ensemble_preds["model_set"] = "ensemble"
    cv_df = pd.concat([cv_df, ensemble_preds], axis=0)


    # rearrange results
    model_results = {}

    for model in t_models:
        model_results[model]= {}
        model_results[model]["rmse"] = []
        model_results[model]["mae"] = [] 

        for fold in range(3):
        
            model_results[model]["mae"].append(results[fold][model]["mae"])
            model_results[model]["rmse"].append(results[fold][model]["rmse"])
        
        model_results[model]["rmse_avg"] = np.array(model_results[model]["rmse"]).mean()
        model_results[model]["mae_avg"] = np.array(model_results[model]["mae"]).mean()

        model_results[model]["rmse_std"] = np.array(model_results[model]["rmse"]).std()
        model_results[model]["mae_std"] = np.array(model_results[model]["mae"]).std()

    for model in t_models:
        print(model)
        print("RMSE ", model_results[model]["rmse_avg"])
        print("MAE ",model_results[model]["mae_avg"])
        print("\n")

        print("RMSE STD ", model_results[model]["rmse_std"])
        print("MAE STD",model_results[model]["mae_std"])
        print("\n")

    return cv_df, model_results

def full_step_2(cv_df:pd.DataFrame,
                experiment_config:dict) -> None:
    
    ANALYSIS_POSTFIX = experiment_config["ANALYSIS_POSTFIX"]
    # TRAIN ON ALL PREDICTIONS AT ONCE

    t_models = ["lr", "svm", "lgbm", "catboost"]

    # Prepare the input data
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(cv_df.loc[cv_df.model_set!="ensemble", "input_sequence"])
    X_train_column_sparse = pd.get_dummies(cv_df.loc[cv_df.model_set!="ensemble", "model_set"], sparse=True).sparse.to_coo().tocsr()
    X_train = hstack([X_train_column_sparse, X_train_tfidf])
    y_train = cv_df.loc[cv_df.model_set!="ensemble", "rouge"]
        
    with open(f"./models/vectorizer_{ANALYSIS_POSTFIX}.pkl", "wb") as file:
        pickle.dump(vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL) 
        
    for model in t_models:
        print(model)
        preds_df = step_two(experiment_config=experiment_config,
                            X_train=X_train,
                            y_train=y_train,
                            model=model,
                            save=True)
        


def test_training_epochs_sets(experiment_config:dict,
                            test_df: pd.DataFrame,
                            test_data: Dataset,
                            train_data: Dataset,
                            tokenizer: AutoTokenizer) -> pd.DataFrame:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rouge = evaluate.load('rouge')

    FULL_TRAIN_ARGS = experiment_config["FULL_TRAIN_ARGS"]
    MODEL_NAME = experiment_config["MODEL_NAME"]
    

    #### PREPARE THE RESULTS DICTIONARY
    results = {}
    for epoch_i, epoch_set in enumerate(sorted(FULL_TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"])):
        
        set_df = test_df.copy()
        print(f"TRAINING EPOCH SET {epoch_set}")

        TRAIN_ARGS = copy.deepcopy(FULL_TRAIN_ARGS)
        MODEL_PATH = f"./models/{epoch_set}_epoch_set"

        results[epoch_set] = {}

        if epoch_set > 1: 
            TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"] = epoch_set - latest_run_epoch
        else:
            TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"] = epoch_set
        
        print(f'TRAINING EPOCHS {TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"]}')

        if epoch_set > 1: 
            MODEL_NAME = f"./models/{latest_run_epoch}_epoch_set"
        
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        print(device)
        model.to(device)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        compute_metrics = ev.compute_metric_with_params(tokenizer) 

        if not os.path.exists(f'reports/'): 
            os.mkdir(f'reports/')

        training_args = Seq2SeqTrainingArguments(
                **TRAIN_ARGS["SEQ_TRAINER_ARGS"],
            )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        if epoch_set!=0:
            trainer.train()

        text = list(test_df["input_sequence"].values)
        summaries = infer.generate_summary(text, model, tokenizer, TRAIN_ARGS["ENCODER_LENGTH"], TRAIN_ARGS["DECODER_LENGTH"])
        
        
        set_df["epoch_set"] = epoch_set
        set_df["prediction"] = summaries[1]
        set_df["rouge"] = rouge.compute(predictions=set_df["prediction"], 
                    references=set_df["output_sequence"],
                    use_stemmer=True, 
                    use_aggregator=False,
                    rouge_types=["rouge1"])["rouge1"]

        if epoch_set==0:
            test_result_df = set_df.copy()
        else: 
            test_result_df = pd.concat([test_result_df, set_df])
        
        ########## SAVE EPOCH SET MODEL
        if not os.path.exists(MODEL_PATH): 
            os.mkdir(MODEL_PATH)

        trainer.save_model(MODEL_PATH)

        latest_run_epoch = epoch_set

    return test_result_df


def test_cluster_set(experiment_config:dict,
                    test_df: pd.DataFrame,
                    test_data: Dataset,
                    tokenizer: AutoTokenizer,
                    results: dict,
                    cluster_id: int,) -> pd.DataFrame:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rouge = evaluate.load('rouge')

    FULL_TRAIN_ARGS = experiment_config["FULL_TRAIN_ARGS"]
    MODEL_NAME = experiment_config["MODEL_NAME"]
    CLUSTER_EPOCHS = experiment_config["CLUSTER_EPOCHS"]
    DATE_STR = experiment_config["DATE_STR"]
    RS = experiment_config["RS"]

    #### LOAD CLUSTER_ID 
    # [7942]
    dataset = pd.read_csv(f"../data/processed/conala/{DATE_STR}/conala_mined_clustered.csv")
    fold_train = dataset.loc[(dataset.cluster==cluster_id) & (~ dataset.question_id.isin(test_df.question_id)),:] 
    fold_train = Dataset.from_pandas(fold_train.sample(n=7942, random_state=RS).reset_index(drop=True))
    fold_train = pr.preprocess_dataset(fold_train, tokenizer=tokenizer, intent_colum_name="intent")

    #### PREPARE THE RESULTS DICTIONARY
    results[f"cluster_{cluster_id}"] = {}

    #### Learning LOOP
    print(f"TRAINING CLUSTER SET {cluster_id} FOR EPOCHS{CLUSTER_EPOCHS}")

    TRAIN_ARGS = copy.deepcopy(FULL_TRAIN_ARGS)
    MODEL_PATH = f"./models/cluster_id{cluster_id}"

    TRAIN_ARGS["SEQ_TRAINER_ARGS"]["num_train_epochs"] = CLUSTER_EPOCHS
            
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print(f"LOADING MODEL {MODEL_NAME}")

    print(device)
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = ev.compute_metric_with_params(tokenizer) 

    if not os.path.exists(f'reports/'): 
        os.mkdir(f'reports/')

    training_args = Seq2SeqTrainingArguments(
            **TRAIN_ARGS["SEQ_TRAINER_ARGS"],
        )
        
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=fold_train,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    ########## SAVE CLUSTER MODEL
    if not os.path.exists(MODEL_PATH): 
        os.mkdir(MODEL_PATH)

    trainer.save_model(MODEL_PATH)

    text = test_data["input_sequence"]
    test_df["prediction"] = generate_summaries_batches(text=text,
                                                        model=model, 
                                                        tokenizer=tokenizer,
                                                        TRAIN_ARGS=TRAIN_ARGS)
    


    test_df["rouge"] = rouge.compute(predictions=test_df["prediction"], 
                        references=test_df["output_sequence"],
                        use_stemmer=True, 
                        use_aggregator=False,
                        rouge_types=["rouge1"])["rouge1"]
            
    results[f"cluster_{cluster_id}"] = test_df
            
    return results