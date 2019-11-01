import fire
import pandas as pd
import yaml
import os

from sklearn.metrics import mean_squared_error

from . import estimator


def main(data_path, params_path, model_name, cache_data=False, toy=False):

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # load data
    if not cache_data or not os.path.exists(
        data_path.replace(".csv", "_train.feather")
    ):
        df = estimator.get_data(data_path, params)

        # preprocessors
        preprocessors = estimator.get_preprocessors(df, params)

        # preprocess
        df = estimator.input_fn(df, params, preprocessors, "train")
        # X_train = estimator.input_fn(X_train, params, preprocessors, "train")
        # X_test = estimator.input_fn(X_test, params, preprocessors, "eval")

        # split
        X_train, X_test = estimator.split(df, params)

        if cache_data:
            X_train.to_feather(data_path.replace(".csv", "_train.feather"))
            X_test.to_feather(data_path.replace(".csv", "_test.feather"))
    else:
        X_train = pd.read_feather(data_path.replace(".csv", "_train.feather"))
        X_test = pd.read_feather(data_path.replace(".csv", "_test.feather"))

    if toy:
        X_train = X_train[:200]

    y_train = X_train.pop(params["label_name"])
    y_test = X_test.pop(params["label_name"])

    # create model
    model = estimator.get_model(model_name, params)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Training Started...")
    model.fit(X_train, y_train)

    print("Calculating Score...")
    score = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)

    print(f"R2: {score}, MSE: {mse}")


if __name__ == "__main__":
    fire.Fire(main)
