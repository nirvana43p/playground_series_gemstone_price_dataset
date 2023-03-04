import warnings

import sys
import os
from os.path import join
import joblib

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,RepeatedKFold, cross_val_score, train_test_split
import optuna

import logging

import multiprocessing


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
MODEL_TYPE = "xgboost_model"

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


def find_best_hyperparameter(X_train, y_train,X_test, y_test,alg):
    if alg == "xgboost":
        def objective(trial):
            """Define the objective function"""

            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
                'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                'eval_metric': 'rmse',
            }

            # Fit the model
            optuna_model = XGBRegressor(**params)
            optuna_model.fit(X_train, y_train)

            y_pred = optuna_model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred)
            return rmse
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15)
        trial = study.best_trial
        return trial.params


if __name__ == "__main__":
    logging.info("Loading Data...")
    df_train, select_feat = read_data()
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET,"id"])[select_feat], df_train[TARGET], test_size=0.3, random_state=SEED)
    #X_train, y_train = df_train.drop(columns=[TARGET,"id"])[select_feat].copy(), df_train[TARGET].copy()
    logger.info("Hyperparameter tunning")
    best_params = find_best_hyperparameter(X_train, y_train, X_test, y_test, alg = "xgboost")
    logger.info("Performance estimation")
    best_score = np.mean(cross_val_score(XGBRegressor(**best_params), X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"))
    logger.info("RMSE perfomance : {}".format(-1 * best_score))
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_metric="rmse", verbose=True)
    logger.info("Generating report..")
    save_simple_metrics_report(-1*best_score, best_params, model_name = MODEL_TYPE, SEED = SEED)
    logger.info("Saving the model..")
    joblib.dump(model,join("model","{}.joblib".format(MODEL_TYPE)))