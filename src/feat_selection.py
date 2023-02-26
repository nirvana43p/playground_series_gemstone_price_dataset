import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

from feature_engine.selection import DropCorrelatedFeatures

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
        Feature selection step:

            - Drop very hight correleted features
    """
    X_train, y_train = df_train.drop(columns=[TARGET,"id"]).copy(), df_train[TARGET].copy()
    logger.info("Feature selection - drop correlated features (0.99 of threshold)")
    drop_corr_transformer =  DropCorrelatedFeatures(threshold=0.99)
    select_feat = drop_corr_transformer.fit_transform(X_train).columns.to_list()
    logger.info(f"Number of features : {X_train.shape[1]}")
    logger.info(f"Number of features selected : {len(select_feat)}")
    pd.Series(select_feat).to_csv(join("data", "selected_feat.csv"), index=False, header = True)


if __name__ == "__main__":
    logging.info("Loading Data...")
    df_train = read_data()
    logger.info("Feature Selection...")
    feature_selection(df_train)
