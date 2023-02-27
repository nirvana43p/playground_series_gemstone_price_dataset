import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

from feature_engine.creation import MathFeatures
from feature_engine.transformation import (
    YeoJohnsonTransformer,
)

from sklearn.preprocessing import QuantileTransformer

import scipy.stats as stats

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

    X_train = X_train.replace("I1","SI2").replace("SI2","SI2andI1")
    X_test = X_test.replace("I1","SI2").replace("SI2","SI2andI1")
    clarity_mappings = {"IF":1, "VVS1":2, "VVS2":3, "VS1":4, "VS2":5,
                    "SI1":6, "SI2andI1":7}
    X_train["clarity"] = X_train["clarity"].map(clarity_mappings)
    X_test["clarity"] = X_test["clarity"].map(clarity_mappings)
    assert ( 
        set(X_test["clarity"].unique()).issubset(set(X_train["clarity"].unique()))
    ), "There are clarity levels missing in test data for mapping"

    return X_train, X_test

def numeric_variable_transform(X_train, X_test):
    X_train["carat"], _ = stats.yeojohnson(X_train["carat"])
    X_test["carat"], _ = stats.yeojohnson(X_test["carat"])
    return X_train, X_test

def agregate_variable_transform(X_train,X_test):
    """
        Agregate functions to enrich features 
    """
    # quality measurement
    quality_vars = ["cut","color","clarity"]
    quality_transformer = MathFeatures(
            variables = quality_vars,
            func = ["mean","prod"])
    X_train = quality_transformer.fit_transform(X_train)
    X_test = quality_transformer.transform(X_test)
    

    # We consider the cubic zirconia as cone
    # Shape measurements of the cubic zirconia as cone
    radious_vars = ["x","y"]
    radious_transformer = MathFeatures(
            variables = radious_vars,
            func = lambda x: 1/4*(x.x + x.y),
            new_variables_names = ["radious"])
    X_train = radious_transformer.fit_transform(X_train)
    X_test = radious_transformer.transform(X_test)


    slant_height_vars = ["radious","z"]
    slant_height_transformer = MathFeatures(
            variables = slant_height_vars,
            func = lambda x:np.sqrt(x.radious**2 + x.z**2),
            new_variables_names = ["slant_height"])
    X_train = slant_height_transformer.fit_transform(X_train)
    X_test = slant_height_transformer.transform(X_test)


    z_ratio_vars = ["radious","z"]
    z_ratio_transformer = MathFeatures(
            variables = z_ratio_vars,
            func = lambda x:x.z/(x.radious + 1e-6),
            new_variables_names = ["z_ratio"])
    X_train = z_ratio_transformer.fit_transform(X_train)
    X_test = z_ratio_transformer.transform(X_test)


    volumne_cone_vars = ["radious","z"]
    volumne_cone_transformer = MathFeatures(
            variables = volumne_cone_vars,
            func = lambda x:1/3*((x.radious**2)*x.z*np.pi),
            new_variables_names = ["cone_volume"])
    X_train = volumne_cone_transformer.fit_transform(X_train)
    X_test = volumne_cone_transformer.transform(X_test)


    area_cone_vars = ["radious","slant_height"]
    area_cone_transformer = MathFeatures(
            variables = area_cone_vars,
            func = lambda x:np.pi*x.radious*(x.radious+x.slant_height),
            new_variables_names = ["cone_area"])
    X_train = area_cone_transformer.fit_transform(X_train)
    X_test = area_cone_transformer.transform(X_test)


    average_girdle_diameter_vars = ["depth","z"]
    average_girdle_diameter_transformer = MathFeatures(
            variables = average_girdle_diameter_vars,
            func = lambda x: x.z/(x.depth + 1e-6),
            new_variables_names = ["average_girdle_diameter"])
    X_train = average_girdle_diameter_transformer.fit_transform(X_train)
    X_test = average_girdle_diameter_transformer.transform(X_test)


    depth_to_table_ratio_vars = ["depth","table"]
    depth_to_table_ratio_transformer = MathFeatures(
            variables = depth_to_table_ratio_vars,
            func = lambda x: x.depth/x.table,
            new_variables_names = ["depth_to_table_ratio"])
    X_train = depth_to_table_ratio_transformer.fit_transform(X_train)
    X_test = depth_to_table_ratio_transformer.transform(X_test)


    table_percentage_vars = ["table","x","y"]
    table_percentage_transformer = MathFeatures(
            variables = table_percentage_vars,
            func = lambda x: (x.table / ((x.x + x.y + 1e-6) / 2)) * 100,
            new_variables_names = ["table_percentage"])
    X_train = table_percentage_transformer.fit_transform(X_train)
    X_test = table_percentage_transformer.transform(X_test)


    depth_percentage_vars = ["depth","x","y"]
    depth_percentage_transformer = MathFeatures(
            variables = depth_percentage_vars,
            func = lambda x: (x.depth / ((x.x + x.y + 1e-6) / 2) + 1e-6) * 100,
            new_variables_names = ["depth_percentage"])
    X_train = depth_percentage_transformer.fit_transform(X_train)
    X_test = depth_percentage_transformer.transform(X_test)


    symmetry_vars = ["x","y","z"]
    symmetry_vars_transformer = MathFeatures(
            variables =  symmetry_vars,
            func = lambda x: (abs(x.x - x.z) + abs(x.y - x.z)) / (x.x + x.y + x.z + 1e-6),
            new_variables_names = ["symmetry"])
    X_train = symmetry_vars_transformer.fit_transform(X_train)
    X_test = symmetry_vars_transformer.transform(X_test)


    average_girdle_diameter_and_cone_area_vars = ["cone_area","average_girdle_diameter"]
    average_girdle_diameter_and_cone_area_transformer = MathFeatures(
            variables = average_girdle_diameter_and_cone_area_vars,
            func = "prod",
            new_variables_names = ["average_girdle_diameter_and_cone_area"])
    X_train = average_girdle_diameter_and_cone_area_transformer.fit_transform(X_train)
    X_test = average_girdle_diameter_and_cone_area_transformer.transform(X_test)


    density_vars = ["carat","cone_volume"]
    density_transformer = MathFeatures(
            variables = density_vars,
            func = lambda x: x.carat*x.cone_volume,
            new_variables_names = ["density"])
    X_train = density_transformer.fit_transform(X_train)
    X_test = density_transformer.transform(X_test)


    # Shape measurements
    carat_and_quality_vars = ["carat","prod_cut_color_clarity"]
    carat_and_quality_transformer = MathFeatures(
            variables = carat_and_quality_vars,
            func = "prod")
    X_train = carat_and_quality_transformer.fit_transform(X_train)
    X_test = carat_and_quality_transformer.transform(X_test)

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

    logger.info("Numeric Variables transformation")
    X_train, X_test = numeric_variable_transform(X_train, X_test)

    logger.info("Agregate variables")
    X_train, X_test = agregate_variable_transform(X_train,X_test)

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
