import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
)
from sklearn.model_selection import train_test_split

from scorepyo._utils import fast_numba_auc
from scorepyo.calibration import VanillaCalibrator
from scorepyo.models import EBMRiskScore
from scorepyo.ranking import LogOddsDensity, MRMRRank


@pytest.mark.parametrize(
    "ranker",
    [LogOddsDensity(), MRMRRank()],
)
def test_end_2_end(ranker):
    data = load_breast_cancer()
    data_X, data_y = data.data, data.target

    X = pd.DataFrame(data=data_X, columns=data.feature_names)
    X["category"] = np.where(X["mean smoothness"] <= 0.1, "A", "B")
    y = pd.Series(data_y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    X_test["category"] = "C"

    min_point_value = -2
    max_point_value = 3
    nb_max_features = 4

    optim_method = fast_numba_auc

    scorepyo_model = EBMRiskScore(
        min_point_value=min_point_value,
        max_point_value=max_point_value,
        nb_max_features=nb_max_features,
        nb_additional_features=6,
        enumeration_maximization_metric=optim_method,
        ranker=ranker,
        calibrator=VanillaCalibrator(),
    )

    scorepyo_model.fit(
        X_train,
        y_train,
        X_calib=None,
        y_calib=None,
        categorical_features=["category"],
    )

    print(scorepyo_model.summary())

    y_proba = scorepyo_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

    precision_recall_curve(y_test.astype(int), y_proba)
    average_precision = np.round(
        average_precision_score(y_test.astype(int), y_proba), 3
    )

    print(f"Average precision : \n{average_precision}")

    precision_test = precision_score(y_test.astype(int), y_proba > 0.5)

    print(f"Precision@0.5: \n{precision_test}")

    assert True
