import pandas as pd
import numpy as np
import pickle 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import hstack

def vectorize_input_df(df: pd.DataFrame,
                       vectorizer: TfidfVectorizer,
                       acc_rouge: float=0.25,
                       fit: bool=True) -> dict:
    
    X = df.loc[:, "input_sequence"]
    if fit:
        X = vectorizer.fit_transform(X)
    else: 
        X = vectorizer.transform(X)

    print(X.shape)

    y = (df.loc[:, "catboost_perf_hat"]>=acc_rouge).astype(int)
    return {"X": X,
            "y": y, 
            "vectorizer": vectorizer}

def train_classifier(X_train: pd.DataFrame,
                     y_train:pd.DataFrame) -> CatBoostClassifier:
    
    classifier = CatBoostClassifier()
    classifier.fit(X=X_train, y=y_train)
    return classifier

def classifier_inference(classifier: CatBoostClassifier,
                         X: pd.DataFrame,
                         vectorizer: TfidfVectorizer): 
    vectorized = vectorize_input_df(df=X, vectorizer=vectorizer, fit=False)
    X = vectorized["X"]

    probs = classifier.predict(X)
    pp = 1 - probs
    probs = np.column_stack((pp, probs))
    return probs