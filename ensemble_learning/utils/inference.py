import pandas as pd
import pickle
from scipy.sparse import hstack
from typing import Tuple
    
def generate_summary(test_samples, model, tokenizer, encoder_max_length, decoder_max_length):

    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=decoder_max_length)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


def generate_summary_fast(model , input_ids, attention_mask, tokenizer, decoder_max_length):

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=decoder_max_length)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str

def pred_perf(experiment_config,
              X,
              model): 

    ANALYSIS_POSTFIX = experiment_config["ANALYSIS_POSTFIX"]

    with open(f'./models/reg_{model}_{ANALYSIS_POSTFIX}.pkl','rb') as f:
            reg = pickle.load(f)

    y_pred = reg.predict(X)
    y_pred[y_pred<0] = 0
    return y_pred


def meta_predict(experiment_config:dict, 
                 test_df: pd.DataFrame,
                 base_models_names: list,
                 t_models:list = ["svm", "catboost"]) -> pd.DataFrame:

    ANALYSIS_POSTFIX = experiment_config["ANALYSIS_POSTFIX"]
    
    for model_i, model_set in enumerate(base_models_names):

        set_df = test_df.copy()
        set_df["model_set"] = model_set
        # Prepare the input data
        with open(f"./models/vectorizer_{ANALYSIS_POSTFIX}.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        if model_i==0:
            meta_preds_df = set_df.copy()
        else: 
            meta_preds_df = pd.concat([meta_preds_df, set_df])
            
    X_test_tfidf = vectorizer.transform(meta_preds_df.loc[:, "input_sequence"])
    X_test_column_sparse = pd.get_dummies(meta_preds_df.loc[:, "model_set"], sparse=True).sparse.to_coo().tocsr()
    X_test = hstack([X_test_column_sparse, X_test_tfidf])

    for model in t_models:
        print(model)
        meta_preds_df[f"{model}_preds"] = pred_perf(experiment_config=experiment_config, 
                                                    X=X_test,
                                                    model=model)

    meta_preds_df = meta_preds_df.reset_index(drop=True)
    return meta_preds_df


def create_ensemble_map(meta_preds_df:pd.DataFrame, t_model_name:str="catboost") -> Tuple:

    models_index = meta_preds_df.groupby("id")[f"{t_model_name}_preds"].idxmax()
    optimal_ensemble = meta_preds_df.iloc[models_index][["id", "model_set"]]
    optimal_ensemble_values = meta_preds_df.iloc[models_index][["id", f"{t_model_name}_preds"]]
    return dict(zip(optimal_ensemble.id, optimal_ensemble.model_set)), optimal_ensemble_values

def ensemble_compute(test_result_df:pd.DataFrame,
                    optimal_ensemble_map:dict) -> pd.DataFrame:

    ### ENSEMBLE COMPUTE
    test_result_df["opt_es_id"] = test_result_df.id.map(optimal_ensemble_map)
    ensemble_preds = test_result_df.loc[test_result_df["model_set"]==test_result_df["opt_es_id"], :]
    ensemble_preds["rouge"].mean()
    ensemble_preds["model_set"] = "ensemble"
    return pd.concat([test_result_df, ensemble_preds], axis=0)