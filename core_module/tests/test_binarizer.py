import warnings

import pytest
from pandera.errors import SchemaError
from sklearn.exceptions import NotFittedError

from scorepyo.binarizers import EBMBinarizer
from scorepyo.exceptions import (
    MissingColumnError,
    NegativeValueError,
    NonBooleanValueError,
    NonIntegerValueError,
)

# from pytest_lazyfixture import lazy_fixture


def test_binarizer_integer_param():
    with pytest.raises(NonIntegerValueError):
        EBMBinarizer(max_number_binaries_by_features=3.5)


def test_binarizer_positive_param():
    with pytest.raises(NegativeValueError):
        EBMBinarizer(max_number_binaries_by_features=-1)


@pytest.mark.parametrize(
    "keep_negative_value",
    [2, "True", "foo", [1, 2], [True], 1, 0],
)
def test_binarizer_boolean_param(keep_negative_value):
    with pytest.raises(NonBooleanValueError):
        EBMBinarizer(keep_negative=keep_negative_value)


def test_missing_categorical_columns(mixed_features, binary_target):
    binarizer = EBMBinarizer()
    with pytest.raises(MissingColumnError):
        binarizer.fit(mixed_features, binary_target, categorical_features=["D"])


def test_missing_to_exclude_columns(mixed_features, binary_target):
    binarizer = EBMBinarizer()
    with pytest.raises(MissingColumnError):
        binarizer.fit(mixed_features, binary_target, to_exclude_features=["D"])


def test_detect_categorical_columns(mixed_features, binary_target):
    binarizer = EBMBinarizer()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
        )
        binarizer.fit(mixed_features, binary_target, categorical_features="auto")

    assert binarizer._categorical_features == ["C"]


def test_not_fitted(mixed_features):
    binarizer = EBMBinarizer()
    with pytest.raises(NotFittedError):
        binarizer.transform(mixed_features)


def test_different_dataframe(mixed_features, continuous_features, binary_target):
    binarizer = EBMBinarizer()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
        )
        binarizer.fit(mixed_features, binary_target, categorical_features="auto")
    with pytest.raises(SchemaError):
        binarizer.transform(continuous_features)


@pytest.mark.parametrize(
    "max_plateau",
    [2, 3, 5, 10],
)
def test_number_plateaus(breast_cancer_features, breast_cancer_target, max_plateau):
    X, y = breast_cancer_features, breast_cancer_target
    binarizer = EBMBinarizer(
        max_number_binaries_by_features=max_plateau, keep_negative=True
    )
    binarizer.fit(X, y)
    X_binarized = binarizer.transform(X)
    assert len(X_binarized.columns) == max_plateau * len(X.columns)


def test_define_categorical_columns(mixed_features, binary_target):
    binarizer = EBMBinarizer()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
        )
        binarizer.fit(mixed_features, binary_target, categorical_features=["C"])
    assert binarizer._categorical_features == ["C"]
    X_binarized = binarizer.transform(mixed_features)
    # test if all categories of C are in the new columns
    assert all([(c in X_binarized.columns) for c in ["C_1", "C_2"]])


@pytest.mark.parametrize(
    "excluded_columns",
    [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "C"], ["A", "B", "C"]],
)
def test_define_exclude_columns(mixed_features, binary_target, excluded_columns):
    binarizer = EBMBinarizer()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
        )
        binarizer.fit(
            mixed_features, binary_target, to_exclude_features=excluded_columns
        )
    print(binarizer._to_exclude_features)
    print(excluded_columns)

    assert binarizer._to_exclude_features == excluded_columns

    X_binarized = binarizer.transform(mixed_features)
    print(X_binarized.columns)
    # test that all columns excluded from binarizer are left as is
    assert all([(c in X_binarized.columns) for c in excluded_columns])

    # test that the new dataframe has the same values as in the original
    assert mixed_features[excluded_columns].equals(X_binarized[excluded_columns])


def test_keep_negative(breast_cancer_features, breast_cancer_target):
    binarizer = EBMBinarizer(keep_negative=False)
    binarizer.fit(breast_cancer_features, breast_cancer_target)
    binarizer.transform(breast_cancer_features)
    df_info = binarizer.get_info()
    print(df_info)
    mask_no_intercept = df_info.index != "intercept"
    mask_nan = ~df_info["log_odds"].isna()
    assert df_info[mask_no_intercept & mask_nan]["log_odds"].astype(float).min() >= 0
