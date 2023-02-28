import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV

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

DATA_TRAIN_PATH = join("data", "train_prepared_feat_eng.csv")

TARGET = "price"

SEED = 8274
np.random.seed(SEED)


def read_data():
    """
        Read train data
    """
    df_train = pd.read_csv(DATA_TRAIN_PATH)

    return df_train

def feature_selection(df_train):
    """
        Feature selection step
    """
    X_train, y_train = df_train.drop(columns=[TARGET,"id"]).copy(), df_train[TARGET].copy()
    logger.info("Feature selection - RFECV")
    min_features_to_select = int(0.5*X_train.shape[1])
    selector = RFECV(XGBRegressor(random_state=SEED), 
                    step=1, cv=5, min_features_to_select=min_features_to_select)
    selector = selector.fit(X_train, y_train)
    logger.info(f"Number of features : {X_train.shape[1]}")
    logger.info(f"Number of features selected : {selector.n_features_}")
    select_feat = X_train.columns[selector.support_]
    pd.Series(select_feat).to_csv(join("data", "selected_feat.csv"), index=False, header = True)


if __name__ == "__main__":
    logging.info("Loading Data...")
    df_train = read_data()
    logger.info("Feature Selection...")
    feature_selection(df_train)
