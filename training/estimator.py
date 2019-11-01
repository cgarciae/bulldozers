import numpy as np
import pandas as pd
from sklearn import preprocessing as sk_preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from .utils import LabelEncoder


def split(df, params):

    X_train, X_test = train_test_split(df, train_size=params["train_size"])

    X_train = X_train.reset_index()
    X_test = X_test.reset_index()

    return X_train, X_test


def get_transform(mode):

    if mode == "train":
        return lambda preprocessor, x: preprocessor.fit_transform(x)
    else:
        return lambda preprocessor, x: preprocessor.transform(x)


def input_fn(df, params, preprocessors, mode):

    df = df.copy()

    transfrom = get_transform(mode)

    ## apply preprocessors
    for column, preprocessor in preprocessors.items():
        try:
            df[column] = transfrom(preprocessor, df[[column]])
        except:
            print(f"Failed on column: {column} asdfsdf")
            raise Exception(f"Failed on column: {column} asdfsdf")

    # drop unused columns
    df.drop(columns=params["date_features"], inplace=True)

    return df


def get_data(data_path, params):

    df = pd.read_csv(data_path, parse_dates=params["date_features"])

    ## data engineering
    df["SalePrice"] = np.log(df["SalePrice"])
    df = df[df.YearMade > 1800]
    df["YearMade"] = df["YearMade"].astype(np.float32)

    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == np.dtype("O"):
            df[col] = df[col].astype(str)

    df.sort_values(by="saledate", inplace=True)

    return df


def get_preprocessors(
    df, params,
):

    return {
        col: Pipeline(
            [
                ("missing", SimpleImputer(strategy="constant", fill_value="nan")),
                ("encode", LabelEncoder()),
            ]
        )
        if df[col].dtype == np.dtype("O")
        else Pipeline(
            [
                ("encode", sk_preprocessing.MinMaxScaler()),
                ("missing", SimpleImputer(strategy="median")),
            ]
        )
        if df[col].dtype == np.float64 or df[col].dtype == np.float32
        else Pipeline(
            [
                ("missing", SimpleImputer(strategy="constant", fill_value=-1)),
                ("encode", LabelEncoder()),
            ]
        )
        for col in df.columns
        if col not in params["date_features"]
    }


def get_model(model_name, params):

    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(**params[model_name])
    else:
        raise ValueError(f"Model not supported: {model_name}")

    return model
