import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
)
from sklearn.model_selection import train_test_split

from scorepyo.binary_featurizer import AutomaticBinaryFeaturizer
from scorepyo.models import OptunaScoreCard


def test_end_2_end():
    # assert True

    data = load_breast_cancer()
    data_X, y = data.data, data.target

    X = pd.DataFrame(data=data_X, columns=data.feature_names)
    X["category"] = np.where(X["mean smoothness"] <= 0.1, "A", "B")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    X_test["category"] = "C"

    binarizer = AutomaticBinaryFeaturizer(
        max_number_binaries_by_features=3, keep_negative=True
    )
    binarizer.fit(
        X_train, y_train, categorical_features="auto", to_exclude_features=None
    )

    X_train_binarized, df_info = binarizer.transform(X_train)
    X_test_binarized, _ = binarizer.transform(X_test)

    # Test without optuna params and with df_info
    scorepyo_model = OptunaScoreCard(
        nb_max_features=4,
        min_point_value=-1,
        max_point_value=2,
        df_info=df_info["feature"].reset_index(),
    )

    scorepyo_model.fit(X_train_binarized, y_train)

    scorepyo_model.summary()

    y_proba = scorepyo_model.predict_proba(X_test_binarized)[:, 1].reshape(-1, 1)

    precision_recall_curve(y_test.astype(int), y_proba)
    average_precision = np.round(
        average_precision_score(y_test.astype(int), y_proba), 3
    )

    print(f"Average precision : \n{average_precision}")

    y_pred_test = scorepyo_model.predict(X_test_binarized)
    precision_test = precision_score(y_test.astype(int), y_pred_test)

    print(f"Precision@0.5: \n{precision_test}")

    # Test with optuna params and no df_info
    scorepyo_model_optuna_params = OptunaScoreCard(
        nb_max_features=4,
        min_point_value=-1,
        max_point_value=2,
        df_info=None,
        optuna_optimize_params={"n_trials": 200, "timeout": 60},
    )

    scorepyo_model_optuna_params.fit(X_train_binarized, y_train)

    scorepyo_model_optuna_params.summary()

    y_proba = scorepyo_model_optuna_params.predict_proba(X_test_binarized)[
        :, 1
    ].reshape(-1, 1)

    precision_recall_curve(y_test.astype(int), y_proba)
    average_precision = np.round(
        average_precision_score(y_test.astype(int), y_proba), 3
    )

    print(f"Average precision: \n{average_precision}")

    y_pred_test = scorepyo_model_optuna_params.predict(X_test_binarized)
    precision_test = precision_score(y_test.astype(int), y_pred_test)

    print(f"Precision@0.5: \n{precision_test}")
    assert True
