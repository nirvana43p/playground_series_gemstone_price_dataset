import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import sys
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


DATA_TRAIN_PATH = join("data", "train_prepared_feat_eng.csv")
DATA_TEST_PATH = join("data", "test_prepared_feat_eng.csv")
SELECTED_FEAT_PATH = join("data", "selected_feat.csv")
MODEL_PATH = join("model", "xgboost_model.joblib")
SUBMISSION_PATH = join("data", "submissions")

TARGET = "price"
MODEL_TYPE = "xgboost_model_tune"


def read_data():
    df_train = pd.read_csv(DATA_TRAIN_PATH)
    df_test = pd.read_csv(DATA_TEST_PATH)
    model = joblib.load(MODEL_PATH)
    select_feat = pd.read_csv(SELECTED_FEAT_PATH)["0"].to_list()


    return df_train, df_test, model, select_feat


def make_submission(X_test, model, submission_idx):
    preds = model.predict(X_test)
    pd.DataFrame({"id": submission_idx, "price": preds}).to_csv(
        join(SUBMISSION_PATH, "{}_submission.csv".format(MODEL_TYPE)), index=False, header=True
    )


if __name__ == "__main__":
    logging.info("Loading Data...")
    df_train, df_test, model, select_feat = read_data()
    X_train, y_train = df_train.drop(columns=TARGET).copy(), df_train[TARGET].copy()

    X_train = X_train[select_feat]
    X_test = df_test[select_feat].copy()
    ids = df_test["id"]
    rmse_train = mean_squared_error(
            y_true = y_train,
            y_pred = model.predict(X_train),
            squared = False
        )
    logger.info(f"RMSE Train : {rmse_train}")
    logger.info(f"Generating submission at {SUBMISSION_PATH}")
    make_submission(X_test, model, ids)