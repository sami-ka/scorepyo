import pytest
from pandera.errors import SchemaError
from sklearn.exceptions import NotFittedError

from scorepyo.binary_featurizer import AutomaticBinaryFeaturizer
from scorepyo.exceptions import NegativeValueError, NonIntegerValueError

# from pytest_lazyfixture import lazy_fixture


def test_binarizer_integer_param():
    with pytest.raises(NonIntegerValueError):
        AutomaticBinaryFeaturizer(max_number_binaries_by_features=3.5)


def test_binarizer_positive_param():
    with pytest.raises(NegativeValueError):
        AutomaticBinaryFeaturizer(max_number_binaries_by_features=-1)


def test_detect_categorical_columns(mixed_features, binary_target):
    binarizer = AutomaticBinaryFeaturizer()
    binarizer.fit(mixed_features, binary_target, categorical_features="auto")

    assert binarizer._categorical_features == ["C"]


def test_not_fitted(mixed_features):
    binarizer = AutomaticBinaryFeaturizer()
    with pytest.raises(NotFittedError):
        binarizer.transform(mixed_features)


def test_different_dataframe(mixed_features, continuous_features, binary_target):
    binarizer = AutomaticBinaryFeaturizer()
    binarizer.fit(mixed_features, binary_target, categorical_features="auto")
    with pytest.raises(SchemaError):
        binarizer.transform(continuous_features)


@pytest.mark.parametrize(
    "max_plateau",
    [2, 3, 5, 10],
)
def test_number_plateaus(breast_cancer_features, breast_cancer_target, max_plateau):
    X, y = breast_cancer_features, breast_cancer_target
    binarizer = AutomaticBinaryFeaturizer(max_number_binaries_by_features=max_plateau)
    binarizer.fit(X, y)
    X_binarized, _ = binarizer.transform(X)
    assert len(X_binarized.columns) == max_plateau * len(X.columns)
