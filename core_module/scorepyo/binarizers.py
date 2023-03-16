"""Class for binarizers:
- EBM-based automatic binarizer
- TODO : Quantile binarizer
"""
import numbers
import time
import warnings
from math import floor, log10
from typing import List, Optional, Protocol, Union

import numpy as np
import pandas as pd
import pandera as pa
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder

from scorepyo.exceptions import (
    MissingColumnError,
    NegativeValueError,
    NonBooleanValueError,
    NonIntegerValueError,
    NumericCheck,
)


class BinarizerProtocol(Protocol):
    """Protocol to respect for future and custom binarizer"""

    def fit(self, X, y, *args, **kwargs):
        ...

    def transform(self, X, **kwargs) -> pd.DataFrame:
        ...

    def get_info(self) -> pd.DataFrame:
        ...


class EBMBinarizer:
    """
    Class for automatic feature binarizer based on EBM.

    This class uses Explainable Boosting Machine (EBM) that are part of the General Additive Model (GAM) family.
    EBM will compute for each feature a tree. The final prediction will be made by summing up the tree value for each feature.
    Since it's single-feature tree, it actually is a piecewise constant scalar function.
    For each interval where the single-feature tree yields a constant contribution for the prediction, this class computes a binary feature.

    Attributes
    ----------
    max_number_binaries_by_features : int
        maximum number of binary features to compute by feature
    keep_negative : bool
        indicator to keep features that decrease predicted probability
    _ebm: ExplainableBoostingClassifier
        EBM classifier that is fitted and used to binarize feature
    _one_hot_encoder: OneHotEncoder
        Scikit-learn one-hot encoder for categorical features
    _categorical_features: list(str)
        list of categorical features
    _continuous_features: list(str)
        list of continuous features
    _to_exclude_features: list(str)
        list of features to exclude from binarizing


    Methods
    -------
    fit(X, y, categorical_features="auto", to_exclude_features=None))
        fits the EBM model on X,y and the one-hot encoder on the categorical features

    transform(X)
        transforms features in X into binarized features based on previously fitted EBM and one-hot encoder
    """

    def __init__(
        self,
        max_number_binaries_by_features: int = 2,
        keep_negative: bool = True,
    ):
        """
        Args:
            max_number_binaries_by_features (int, optional): maximum number of binary features to compute by feature. Defaults to 2.
            keep_negative (bool, optional): indicator to keep features that decrease predicted probability. Defaults to False.
        """

        if max_number_binaries_by_features <= 0:
            raise NegativeValueError(
                f"max_number_binaries_by_features must be a strictly positive integer. \n {max_number_binaries_by_features} is not positive."
            )
        if not isinstance(
            max_number_binaries_by_features,
            numbers.Integral,
        ):
            raise NonIntegerValueError(
                f"max_number_binaries_by_features must be a strictly positive integer. \n {max_number_binaries_by_features} is not an integer."
            )

        self.max_number_binaries_by_features = max_number_binaries_by_features

        if not isinstance(keep_negative, bool):
            raise NonBooleanValueError(
                f"keep_negative attribute must be a boolean. \n {keep_negative} is not a boolean"
            )

        self.keep_negative: bool = keep_negative

        # Creation of underlying EBM to extract binary feature
        # interactions to 0 to prevent pairwise interaction feature
        # TODO : Include pairwise interaction into featurizer?
        # max_bins controls the number of split for each single-feature tree
        self._ebm: BaseEstimator = ExplainableBoostingClassifier(
            interactions=0,
            max_bins=self.max_number_binaries_by_features + 2,
            min_samples_leaf=10,
        )

        # One-hot encoder that imputes infrequent_if_exist for unknown categories
        # it allows only the 10 most frequent categories, in order to not create too many columns
        # for high-cardinality categories
        self._one_hot_encoder: BaseEstimator = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            # max_categories=10,
            sparse=False,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: Union[list[str], str] = "auto",
        to_exclude_features: Optional[list[str]] = None,
    ):
        """Fit function of binarizer

        This functions fits the EBM on X,y and the one-hot encoder on X.
        It stores the categorical, continuous and to exclude columns.

        Args:
            X (pandas.Dataframe): Dataframe of features to use to fit the binarizer
            y (pandas.Series): target values
            categorical_features (list(str), optional): list of categorical features to one-hot encode.
            Defaults to "auto" to automatically detect categorical features.
            to_exclude_features (list(str), optional): List of features to leave as is.
            Defaults to None.
        """
        start_time = time.time()

        # TODO Use pandera
        if categorical_features == "auto":
            self._categorical_features = X.select_dtypes(
                include=[
                    "category",
                    "object",
                    "bool",
                ]
            ).columns
        else:
            not_present_categorical_features = set(categorical_features) - set(
                X.columns
            )
            if len(not_present_categorical_features) > 0:
                raise MissingColumnError(
                    f"{not_present_categorical_features} are not in columns of X."
                )

            self._categorical_features = categorical_features
        if to_exclude_features is None:
            self._to_exclude_features = []
        else:
            not_present_to_exclude_features = set(to_exclude_features) - set(X.columns)
            if len(not_present_to_exclude_features) > 0:
                raise MissingColumnError(
                    f"{not_present_to_exclude_features} are not in columns of X."
                )
            self._to_exclude_features = to_exclude_features

        self._categorical_features = [
            c for c in self._categorical_features if c not in self._to_exclude_features
        ]
        self._continuous_features = [
            c
            for c in X.columns
            if (c not in self._categorical_features)
            and (c not in self._to_exclude_features)
        ]
        cond_list = [
            [c in self._continuous_features for c in X.columns],
            [c in self._categorical_features for c in X.columns],
        ]
        choice_list = ["continuous", "nominal"]
        ebm_feature_type_param = np.select(
            condlist=cond_list, choicelist=choice_list, default=None  # type: ignore[arg-type]
        )
        self._ebm.set_params(feature_types=ebm_feature_type_param)
        self._ebm.fit(X, y)

        if len(self._categorical_features) > 0:
            self._one_hot_encoder.fit(X[self._categorical_features])

        ebm_global = self._ebm.explain_global(name="EBM")

        list_scores: List[Optional[float]] = []
        list_lower_threshold: List[Optional[float]] = []
        list_upper_threshold: List[Optional[float]] = []
        list_category_value: List[Optional[str]] = []
        list_original_column: List[Optional[str]] = []
        list_binary_feature_names: List[Optional[str]] = []
        list_columns = []
        list_feature_type: List[Optional[str]] = []
        list_density: List[Optional[float]] = []

        if len(self._categorical_features) > 0:
            dict_categorical_features = dict()

        # For each continuous feature, we look at their single-feature tree
        for (
            i,
            feature_name,
        ) in enumerate(ebm_global.data()["names"]):
            if feature_name in self._continuous_features:

                # Get EBM info for current feature
                dict_EBM_info_feature = ebm_global.data(i)

                # Get number of plateaus for the corresponding tree
                number_plateau = len(ebm_global.data(i)["scores"])

                # For each plateau, extract lower and upper bound of each interval
                for j in range(number_plateau):
                    contrib = dict_EBM_info_feature["scores"][j]

                    # If logodd contribution of the feature is negative,
                    # check if we should keep negative contribution to probability
                    if (contrib < 0) and (not self.keep_negative):
                        continue

                    # Store lower and upper bound of interval
                    threshold_lower = dict_EBM_info_feature["names"][j]
                    threshold_upper = dict_EBM_info_feature["names"][j + 1]

                    # Round precision for column name
                    # needed when difference upper and lower bound is below 2 digits

                    # try:
                    difference = threshold_upper - threshold_lower
                    # except:
                    #     # Debug
                    #     print(feature_name)
                    #     print(dict_EBM_info_feature)
                    #     assert False

                    needed_round_precision = -floor(log10(abs(difference)))
                    needed_round_precision = max(needed_round_precision, 2)

                    # Case of the first plateau, only have an upper bound for current feature value
                    if j == 0:
                        col_name = f"{feature_name} < {np.round(threshold_upper,needed_round_precision)}"

                        list_columns.append(
                            pd.DataFrame(
                                data=(X[feature_name] < threshold_upper).values,
                                columns=[col_name],
                            )
                        )
                        threshold_lower = np.nan

                    # Case of the last plateau, only have a lower bound for current feature value
                    elif j == number_plateau - 1:
                        col_name = f"{feature_name} >= {np.round(threshold_lower,needed_round_precision)}"
                        list_columns.append(
                            pd.DataFrame(
                                data=(X[feature_name] >= threshold_lower).values,
                                columns=[col_name],
                            )
                        )
                        threshold_upper = np.nan

                    # Case for other inbetween plateaus
                    else:
                        col_name = f"{np.round(threshold_lower,needed_round_precision)} <= {feature_name} < {np.round(threshold_upper,needed_round_precision)}"
                        list_columns.append(
                            pd.DataFrame(
                                data=(
                                    (X[feature_name] < threshold_upper)
                                    & (X[feature_name] >= threshold_lower)
                                ).values,
                                columns=[col_name],
                            )
                        )

                    list_density.append(list_columns[-1][col_name].sum())
                    list_feature_type.append("continuous")

                    list_lower_threshold.append(threshold_lower)
                    list_upper_threshold.append(threshold_upper)

                    list_category_value.append(None)

                    list_scores.append(contrib)
                    list_original_column.append(feature_name)
                    list_binary_feature_names.append(col_name)
            else:
                if (len(self._categorical_features) > 0) and (
                    feature_name in self._categorical_features
                ):
                    # Get EBM info for current feature
                    dict_EBM_info_feature = ebm_global.data(i)

                    # # Get number of plateaus for the corresponding tree
                    # number_categories = len(ebm_global.data(i)["names"])

                    feature_name = ebm_global.data()["names"][i]

                    # list_binary_features_name = []
                    for category_value, score, density in zip(
                        ebm_global.data(i)["names"],
                        ebm_global.data(i)["scores"],
                        ebm_global.data(i)["density"]["scores"],
                    ):
                        # If category value is of type int or float
                        if isinstance(category_value, int) or isinstance(
                            category_value, float
                        ):
                            # Test if int or float
                            if int(category_value) == float(category_value):
                                # it's a int
                                category_value = int(category_value)
                            else:
                                # it's a float
                                category_value = float(category_value)

                        binary_feature_name = "_".join(
                            [str(feature_name), str(category_value)]
                            # [feature_name, str(category_value)]
                        )
                        dict_categorical_features[binary_feature_name] = {
                            "score": score,
                            "density": density,
                        }

        # TODO : Put logodds of ebm for categorical features
        if len(self._categorical_features) > 0:
            # One-hot encode categorical features
            one_hot_categorical_columns = self._one_hot_encoder.get_feature_names_out()
            list_columns.append(
                pd.DataFrame(
                    data=(
                        self._one_hot_encoder.transform(X[self._categorical_features])
                    ),
                    columns=one_hot_categorical_columns,
                )
            )

            list_binary_feature_names.extend(one_hot_categorical_columns)

            # Add info for info dataframe
            for name_out in self._one_hot_encoder.get_feature_names_out():
                list_scores.append(dict_categorical_features[name_out]["score"])
                list_density.append(dict_categorical_features[name_out]["density"])
                list_feature_type.append("categorical")
                list_lower_threshold.append(None)
                list_upper_threshold.append(None)
                list_category_value.append(name_out.split("_")[-1])
                list_original_column.append("_".join(name_out.split("_")[:-1]))

        # Copy features to exclude from binarizing
        if len(self._to_exclude_features) > 0:
            list_columns.append(X[self._to_exclude_features].copy())

            for name_out in self._to_exclude_features:
                list_scores.append(0)
                list_density.append(0)
                list_feature_type.append(None)  # type: ignore
                list_lower_threshold.append(None)
                list_upper_threshold.append(None)
                list_category_value.append(None)
                list_original_column.append(name_out)

        # concat all created columns at the end (vs on the fly) for performance
        X_binarized = pd.concat(list_columns, axis=1)

        # convert to 1/0
        with warnings.catch_warnings():
            # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
            # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
            # See also: https://stackoverflow.com/q/74057367/859591
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=(
                    ".*will attempt to set the values inplace instead of always setting a new array. "
                    "To retain the old behavior, use either.*"
                ),
            )
            X_binarized.loc[:, list_binary_feature_names] = X_binarized.loc[
                :, list_binary_feature_names
            ].astype(int)

        # Regroup information about binary features created in the info dataframe
        # this dataframe contains:
        # - the list for each binary feature of log odds contribution
        # - the list for each binary feature of lower threshold used
        # - the list for each binary feature of upper threshold used
        # - the list for each binary feature of the original feature it came from

        df_score_feature = pd.DataFrame(
            index=X_binarized.columns,
            columns=[
                "log_odds",
                "lower_threshold",
                "upper_threshold",
                "category_value",
                "feature",
                "type",
                "density",
            ],
            data=np.array(
                [
                    list_scores,
                    list_lower_threshold,
                    list_upper_threshold,
                    list_category_value,
                    list_original_column,
                    list_feature_type,
                    list_density,
                ]
            ).T,
        )

        # the intercept is the basis log odd value of the EBM without adding/substracting
        # the different log odds contribution of each feature
        row_intercept = pd.DataFrame(
            index=["intercept"],
            columns=df_score_feature.columns,
            data=[[self._ebm.intercept_[0], None, None, None, "intercept", None, None]],
        )
        df_score_feature = pd.concat(
            [
                df_score_feature,
                row_intercept,
            ],
            axis=0,
        )
        df_score_feature.index.name = "binary_feature"

        self.df_score_feature = df_score_feature.copy()
        self._fit_time = time.time() - start_time
        # return (
        #     X_binarized,
        #     df_score_feature,
        # )

    # TODO : Provide function that give the quality of the binarizer via the learning metrics of ebm

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self._ebm, "has_fitted_", False):
            raise NotFittedError("AutomaticFeatureBinarizer has not been fitted.")

        dict_check_continuous_features = {
            c: pa.Column(checks=[NumericCheck()]) for c in self._continuous_features
        }

        dict_check_other_features = {
            c: pa.Column()
            for c in self._categorical_features + self._to_exclude_features
        }

        dataframe_schema = pa.DataFrameSchema(
            {
                **dict_check_continuous_features,
                **dict_check_other_features,
            },
            strict=True,  # Enable check of other columns in the dataframe
        )

        dataframe_schema.validate(X)

        list_columns = []
        list_binary_feature_names = []
        for (
            binary_feature_name,
            row,
        ) in self.df_score_feature.iterrows():
            feature_name = row["feature"]
            lower_threshold = row["lower_threshold"]
            upper_threshold = row["upper_threshold"]
            # category_value = row["category_value"]
            type = row["type"]

            if type == "continuous":
                list_binary_feature_names.append(binary_feature_name)
                if lower_threshold is np.nan:
                    list_columns.append(
                        pd.DataFrame(
                            data=(X[feature_name] < upper_threshold).values,
                            columns=[binary_feature_name],
                        )
                    )
                elif upper_threshold is np.nan:
                    list_columns.append(
                        pd.DataFrame(
                            data=(X[feature_name] >= lower_threshold).values,
                            columns=[binary_feature_name],
                        )
                    )
                else:
                    list_columns.append(
                        pd.DataFrame(
                            data=(
                                (X[feature_name] < upper_threshold)
                                & (X[feature_name] >= lower_threshold)
                            ).values,
                            columns=[binary_feature_name],
                        )
                    )

        if len(self._categorical_features) > 0:
            # One-hot encode categorical features
            one_hot_categorical_columns = self._one_hot_encoder.get_feature_names_out()
            list_columns.append(
                pd.DataFrame(
                    data=(
                        self._one_hot_encoder.transform(X[self._categorical_features])
                    ),
                    columns=one_hot_categorical_columns,
                )
            )

            list_binary_feature_names.extend(one_hot_categorical_columns)

        # Copy features to exclude from binarizing
        if len(self._to_exclude_features) > 0:
            list_columns.append(X[self._to_exclude_features].copy())

        # concat all created columns at the end (vs on the fly) for performance
        X_binarized = pd.concat(list_columns, axis=1)

        # convert to 1/0
        with warnings.catch_warnings():
            # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
            # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
            # See also: https://stackoverflow.com/q/74057367/859591
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=(
                    ".*will attempt to set the values inplace instead of always setting a new array. "
                    "To retain the old behavior, use either.*"
                ),
            )
            X_binarized.loc[:, list_binary_feature_names] = X_binarized.loc[
                :, list_binary_feature_names
            ].astype(int)

            return X_binarized

    def get_info(self) -> pd.DataFrame:
        return self.df_score_feature
