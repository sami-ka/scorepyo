import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer


@pytest.fixture(scope="module")
def continuous_features():
    X = pd.DataFrame(columns=["A", "B"], data=[[0.2, 0.7], [0.1, 0.9], [7.8, -7]])
    return X


@pytest.fixture(scope="module")
def mixed_features():
    X = pd.DataFrame(
        columns=["A", "B", "C"], data=[[0.2, 0.7, "1"], [0.1, 0.9, "2"], [7.8, -7, "2"]]
    )
    return X


@pytest.fixture(scope="module")
def continuous_target():
    y = pd.Series(data=[0.2, 0.7, 0.5])
    return y


@pytest.fixture(scope="module")
def binary_target():
    y = pd.Series(data=[0, 1, 1])
    return y


@pytest.fixture(scope="module")
def multiclass_target():
    y = pd.Series(data=[0, 1, 2])
    return y


@pytest.fixture(scope="module")
def breast_cancer_dataset():
    data = load_breast_cancer()
    return data


@pytest.fixture(scope="module")
def breast_cancer_features(breast_cancer_dataset):
    data = breast_cancer_dataset
    data_X = data.data
    X = pd.DataFrame(data=data_X, columns=data.feature_names)
    return X


@pytest.fixture(scope="module")
def breast_cancer_target(breast_cancer_dataset):
    data = breast_cancer_dataset
    return data.target
