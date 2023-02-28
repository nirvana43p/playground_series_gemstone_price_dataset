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
from sklearn.model_selection import GridSearchCV,RepeatedKFold, cross_val_score

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

def hyperparameters_tunne(set_tune):
    def hyperparameters_tunned(xgb, X, y, param_grid, scoring, n_splits_cv, **kwargs):
        set_tune(xgb, X, y, param_grid, scoring, n_splits_cv, **kwargs)
        grid = GridSearchCV(
            estimator=xgb.set_params(**kwargs),
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=multiprocessing.cpu_count() - 1,
            cv=RepeatedKFold(n_splits=n_splits_cv, n_repeats=1, random_state=SEED),
            refit=True,
            verbose=0,
            return_train_score=True,
        )
        grid.fit(X=X, y=y)
        best_params = {param: grid.best_params_[param] for param in param_grid.keys()}
        for param, value in best_params.items():
            logger.info("Best {0} : {1}".format(param, value))
        return best_params

    return hyperparameters_tunned


@hyperparameters_tunne
def subsample_colsample_tunne(xgb, X, y, param_grid, scoring, n_splits_cv, gamma=0.0):
    logger.info("Optimizing subsample and colsample_bytree")


@hyperparameters_tunne
def gamma_tunne(xgb, X, y, param_grid, scoring, n_splits_cv, max_depth=5, min_child_weight=3):
    logger.info("Optimizing gamma")


@hyperparameters_tunne
def maxd_mincw_tunne(xgb, X, y, param_grid, scoring, n_splits_cv, n_estimators=200):
    logger.info("Optimizing max_depth and min_child_weight")


@hyperparameters_tunne
def no_trees_tunne(xgb, X, y, param_grid, scoring, n_splits_cv, learning_rate=0.001):
    logger.info("Optimizing number of trees")


def lr_tunne(xgb, X, y, param_grid, eval_metric, scoring, n_experiments=3, n_splits_cv=7):
    logger.info("Optimizing learning rate")
    # loop over n experiments val set to test early stopping
    best_learning_rates = []
    for no_experiment in range(n_experiments):
        idx_val = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.1), replace=False)
        X_val = X.iloc[idx_val, :].copy().reset_index(drop=True)
        y_val = y.iloc[idx_val].copy().reset_index(drop=True)
        X_train = X.reset_index(drop=True).drop(idx_val, axis=0).copy()
        y_train = y.reset_index(drop=True).drop(idx_val, axis=0).copy()

        fit_params = {
            "early_stopping_rounds": 10,
            "eval_metric": eval_metric,
            "eval_set": [(X_val, y_val)],
            "verbose": 0,
        }
        grid = GridSearchCV(
            estimator=xgb.set_params(n_estimators=1000),
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=multiprocessing.cpu_count() - 1,
            cv=RepeatedKFold(n_splits=n_splits_cv, n_repeats=1, random_state=SEED),
            refit=True,
            verbose=0,
            return_train_score=True,
        )
        grid.fit(X=X_train, y=y_train, **fit_params)
        best_lr = grid.best_params_["learning_rate"]
        best_learning_rates.append(best_lr)
        no_trees = len(grid.best_estimator_.get_booster().get_dump())
        logger.info("Best lr: {0}. Number of experiment: {1}".format(best_lr, no_experiment + 1))
        logger.info("Number of trees included in the model: {}".format(no_trees))

    return {"learning_rate": sum(best_learning_rates) / len(best_learning_rates)}


def xgboost_hyper_tunne(task, X, y):
    best_params = {}

    if task == "regression":
        xgbr = XGBRegressor(seed=SEED)
        eval_metric = "rmse"  # for early stopping
        scoring = "neg_root_mean_squared_error"  # for CV
        n_splits_cv = 2
        params_grid = {
            "subsample_colsample": {
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            },
            "gamma": {"gamma": [i / 10.0 for i in range(5)]},
            "max_depth_min_child_weight": {
                "max_depth": [None, 3, 5, 10, 20],
                "min_child_weight": [None, 1, 3, 5],
            },
            "n_estimators": {"n_estimators": range(100, 500, 50)},
            "learning_rate": {"learning_rate": np.linspace(0.001, 0.1, num=10)},
        }

    best_lr = lr_tunne(
        xgbr,
        X,
        y,
        params_grid["learning_rate"],
        eval_metric,
        scoring,
        n_experiments=1,
        n_splits_cv=n_splits_cv,
    )
    best_params.update(best_lr)
    best_no_estimators = no_trees_tunne(
        xgbr,
        X,
        y,
        params_grid["n_estimators"],
        scoring,
        n_splits_cv,
        learning_rate=best_lr["learning_rate"],
    )
    best_params.update(best_no_estimators)
    best_max_depth_min_child_weight = maxd_mincw_tunne(
        xgbr,
        X,
        y,
        params_grid["max_depth_min_child_weight"],
        scoring,
        n_splits_cv,
        n_estimators=best_no_estimators["n_estimators"],
    )
    best_params.update(best_max_depth_min_child_weight)
    best_gamma = gamma_tunne(
        xgbr,
        X,
        y,
        params_grid["gamma"],
        scoring,
        n_splits_cv,
        max_depth=best_max_depth_min_child_weight["max_depth"],
        min_child_weight=best_max_depth_min_child_weight["min_child_weight"],
    )
    best_params.update(best_gamma)
    best_subsample_colsample = subsample_colsample_tunne(
        xgbr,
        X,
        y,
        params_grid["subsample_colsample"],
        scoring,
        n_splits_cv,
        gamma=best_gamma["gamma"],
    )
    best_params.update(best_subsample_colsample)
    # Setting best_subsample, best_colsample_bytree
    xgbr.set_params(**best_subsample_colsample)
    best_score = np.mean(cross_val_score(xgbr, X, y, cv=3, scoring=scoring))
    return xgbr, best_score, best_params


if __name__ == "__main__":
    logging.info("Loading Data...")
    df_train, select_feat = read_data()
    X_train, y_train = df_train.drop(columns=[TARGET,"id"])[select_feat].copy(), df_train[TARGET].copy()
    logger.info("Hyperparameter tunning")
    xgbr, best_score_tunned, best_params = xgboost_hyper_tunne("regression", X_train, y_train)
    logger.info("{}".format(xgbr))
    logger.info("RMSE of tunning : {}".format(-1 * best_score_tunned))
    logger.info("Performance estimation")
    best_score = np.mean(cross_val_score(xgbr, X_train, y_train, cv=10, scoring="neg_root_mean_squared_error"))
    logger.info("RMSE perfomance : {}".format(-1 * best_score))
    model = XGBRegressor(**xgbr.get_params())
    logger.info("Training..")
    model.fit(X_train, y_train, eval_metric="rmse", verbose=True)
    logger.info("Generating report..")
    save_simple_metrics_report(best_score, best_params, model_name = MODEL_TYPE, SEED = SEED)
    logger.info("Saving the model..")
    joblib.dump(model,join("model","{}.joblib".format(MODEL_TYPE)))