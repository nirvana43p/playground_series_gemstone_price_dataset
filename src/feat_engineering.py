import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

import sys
import os
from os.path import join

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DATA_TRAIN_PATH = join("data", "train.csv")
DATA_TEST_PATH = join("data", "test.csv")

TARGET = "price"

SEED = 8274
np.random.seed(SEED)

def read_data():
    """
        Read train and test data
    """
    df_train = pd.read_csv(DATA_TRAIN_PATH)
    df_test = pd.read_csv(DATA_TEST_PATH)

    return df_train, df_test

def categorical_variable_transform(X_train, X_test):
    """
        Map categorical variables to integer vvariables based on level
    """
    color_mappings = {"D":1, "E":2, "F":3, "G":4, "H":5, "I":6, "J":7}
    X_train["color"] = X_train["color"].map(color_mappings)
    X_test["color"] = X_test["color"].map(color_mappings)

    assert ( 
        set(X_test["color"].unique()).issubset(set(X_train["color"].unique()))
    ), "There are colors leves missing in test data for mapping"

    cut_mappings = {"Fair":1, "Good":2, "Very Good":3, "Premium":5, "Ideal":6}
    X_train["cut"] = X_train["cut"].map(cut_mappings)
    X_test["cut"] = X_test["cut"].map(cut_mappings)
    assert ( 
        set(X_test["cut"].unique()).issubset(set(X_train["cut"].unique()))
    ), "There are cut levels missing in test data for mapping"

    clarity_mappings = {"IF":1, "VVS1":2, "VVS2":3, "VS1":4, "VS2":5,
                    "SI1":6, "SI2":7, "I1":8}
    X_train["clarity"] = X_train["clarity"].map(clarity_mappings)
    X_test["clarity"] = X_test["clarity"].map(clarity_mappings)
    assert ( 
        set(X_test["clarity"].unique()).issubset(set(X_train["clarity"].unique()))
    ), "There are clarity levels missing in test data for mapping"

    return X_train, X_test

def feature_engineering(df_train, df_test):
    """
        Make feature engineering step

            - Transform categorical variables to integer variables
    """
    X_train, y_train = df_train.drop(columns = [TARGET]).copy(), df_train[TARGET].copy()
    X_test = df_test.copy()
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "train and test does not have the same number of features"

    logger.info("Categorical Variables Transfomrmation")
    X_train, X_test = categorical_variable_transform(X_train, X_test)

    logger.info("Saving feature engineering and processing data")
    pd.concat([X_train,y_train], axis = 1).to_csv(join("data","train_prepared_feat_eng.csv"),
        index = False, header = True)
    X_test.to_csv(join("data","test_prepared_feat_eng.csv"),index = False, header = True)


if __name__ == "__main__":
    logging.info("Loading Data...")
    df_train, df_test = read_data()
    logging.info("Feature Engineering starting...")
    feature_engineering(df_train, df_test)
    logging.info("Feature Engineering Finished...")   
