# import warnings

import pytest
from pandera.errors import SchemaError
from sklearn.exceptions import NotFittedError

from scorepyo.exceptions import (
    MinPointOverMaxPointError,
    NegativeValueError,
    NonIntegerValueError,
)
from scorepyo.models import OptunaRiskScore, _BaseRiskScore


def test_binarizer_nb_max_features():
    with pytest.raises(NegativeValueError):
        OptunaRiskScore(nb_max_features=-1)

    with pytest.raises(NonIntegerValueError):
        OptunaRiskScore(nb_max_features=1.2)


def test_min_point_value():
    with pytest.raises(NonIntegerValueError):
        OptunaRiskScore(min_point_value=1.2)

    with pytest.raises(NonIntegerValueError):
        OptunaRiskScore(min_point_value=-2.7)


def test_max_point_value():
    with pytest.raises(NonIntegerValueError):
        OptunaRiskScore(max_point_value=1.2)

    with pytest.raises(NonIntegerValueError):
        OptunaRiskScore(max_point_value=-2.7)


def test_min_over_max_point_value():
    with pytest.raises(MinPointOverMaxPointError):
        OptunaRiskScore(min_point_value=3, max_point_value=1)


@pytest.mark.parametrize(
    "erroneous_column_names",
    [
        ["confeature", "binary_feature"],
        ["feature", "binary_confeature"],
        ["confeature", "binary_confeature"],
    ],
)
def test_schema_df_info(df_info_binary_features, erroneous_column_names):
    erroneous_df_info = df_info_binary_features.copy()
    erroneous_df_info.columns = erroneous_column_names
    with pytest.raises(SchemaError):
        OptunaRiskScore(df_info=erroneous_df_info)


def test_base_score_card_fit(binary_features, binary_target):
    with pytest.raises(NotImplementedError):
        score_card = _BaseRiskScore()
        score_card.fit(binary_features, binary_target)


def test_optuna_score_card_predict(binary_features):
    with pytest.raises(NotFittedError):
        score_card = OptunaRiskScore()
        score_card.predict(binary_features)


def test_optuna_score_card_predict_proba(binary_features):
    with pytest.raises(NotFittedError):
        score_card = OptunaRiskScore()
        score_card.predict_proba(binary_features)


def test_optuna_score_card_summary():
    with pytest.raises(NotFittedError):
        score_card = OptunaRiskScore()
        score_card.summary()
