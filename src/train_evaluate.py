import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold, cross_val_score


import sys
import os
from os.path import join
import joblib

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from utils import save_simple_metrics_report

DATA_TRAIN_PATH = join("data", "train_prepared_feat_eng.csv")
SELECTED_FEAT_PATH = join("data", "selected_feat.csv")
MODEL_TYPE = "baseline_model"

TARGET = "price"

SEED = 8274
np.random.seed(SEED)


def read_data():
    """
        Read train data and selected features
    """
    df_train = pd.read_csv(DATA_TRAIN_PATH)
    select_feat = pd.read_csv(SELECTED_FEAT_PATH)["0"].to_list()

    return df_train, select_feat

def feature_scaling(X_train, select_feat):
    """
        Scale data
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train[select_feat])

    X_train_scaled = pd.DataFrame(scaler.transform(X_train[select_feat]), columns=select_feat)
    joblib.dump(scaler, join("model", "minmax_scaler.joblib"))

    return X_train_scaled


def find_best_alpha(model):
    """
        Find best alpha based con CV
    """
    logger.info("Finding best alpha...")
    mse_cv = model.mse_path_.mean(axis=1)
    rmse_cv = np.sqrt(mse_cv)

    best_alpha = model.alphas_[np.argmin(rmse_cv)]
    logger.info(f"Best alpha value found : {best_alpha}")

    return best_alpha


def lineal_model_hyper_tunne(X, y):
    """
        Tune alpha of a lineal model
    """
    best_params = {}

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    alphas = np.logspace(-10, 3, 200)
    model = LassoCV(alphas=alphas, cv=cv)
    _ = model.fit(X, y)
    best_alpha = find_best_alpha(model)
    best_params["alpha"] = best_alpha

    return best_params

def lineal_model_training(best_params, X, y):
    """
        Train and evaluate a lineal model
    """
    best_score = np.mean(
        cross_val_score(
            Lasso(**best_params, random_state=SEED),
            X,
            y,
            cv=RepeatedKFold(n_splits=15, n_repeats=1, random_state=SEED),
            scoring="neg_root_mean_squared_error",
        )
    )
    logger.info("RMSE perfomance : {}".format(-1 * best_score))
    model = Lasso(**best_params, random_state=SEED).fit(X, y)
    logger.info("Training Finished...")
    return model, -1 * best_score

if __name__ == "__main__":
    logging.info("Loading Data...")
    df_train, select_feat = read_data()
    X_train, y_train = df_train.drop(columns=["price","id"]).copy(), df_train["price"].copy()
    logger.info("Scaling the data")
    X_train_scaled = feature_scaling(X_train, select_feat)
    logger.info("Hyperparameter tunning...")
    best_params = lineal_model_hyper_tunne(X_train_scaled, y_train)
    logger.info("Training...")
    model, eval_score = lineal_model_training(best_params, X_train_scaled, y_train)
    logger.info("Generating Report...")
    save_simple_metrics_report(eval_score, best_params, model_name = MODEL_TYPE, SEED = SEED)
    logger.info("Saving Model...")
    joblib.dump(model,join("model","baseline_model.joblib"))