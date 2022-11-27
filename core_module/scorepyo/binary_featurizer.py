"""
TODO Add typing
TODO eneble possibility of regressor?

Returns:

    _type_: _description_
"""
import numbers
import warnings

import numpy as np
import pandas as pd
import pandera as pa
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from scorepyo.exceptions import NegativeValueError, NonIntegerValueError, NumericCheck


class AutomaticBinaryFeaturizer:
    """
    Class for automatic binary featurizer.

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
        max_number_binaries_by_features=2,
        keep_negative=True,
    ):
        """
        Args:
            max_number_binaries_by_features (int, optional): maximum number of binary features to compute by feature. Defaults to 2.
            keep_negative (bool, optional): indicator to keep features that decrease predicted probability. Defaults to False.
        """

        if max_number_binaries_by_features <= 0:
            raise scorepyo.exceptions.NegativeValueError(
                f"max_number_binaries_by_features must be a strictly positive integer. \n {max_number_binaries_by_features} is not positive."
            )
        if not isinstance(max_number_binaries_by_features, numbers.Integral):
            raise NonIntegerValueError(
                f"max_number_binaries_by_features must be a strictly positive integer. \n {max_number_binaries_by_features} is not an integer."
            )

        self.max_number_binaries_by_features = max_number_binaries_by_features

        self.keep_negative = keep_negative

        # Creation of underlying EBM to extract binary feature
        # interactions to 0 to prevent pairwise interaction feature
        # TODO : Include pairwise interaction into featurizer?
        # max_bins controls the number of split for each single-feature tree
        self._ebm = ExplainableBoostingClassifier(
            interactions=0, max_bins=self.max_number_binaries_by_features + 2
        )

        # One-hot encoder that imputes infrequent_if_exist for unknown categories
        # it allows only the 10 most frequent categories, in order to not create too many columns
        # for high-cardinality categories
        self._one_hot_encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", max_categories=10, sparse=False
        )

    def fit(self, X, y, categorical_features="auto", to_exclude_features=None):
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

        # TODO Use pandera
        if categorical_features == "auto":
            self._categorical_features = X.select_dtypes(
                include=["category", "object", "bool"]
            ).columns
        else:
            not_present_categorical_features = set(categorical_features) - set(
                X.columns
            )
            if len(not_present_categorical_features) > 0:
                raise warnings.warn(
                    f"{not_present_categorical_features} are not in columns of X."
                )

            self._categorical_features = categorical_features
        if to_exclude_features is None:
            self._to_exclude_features = []
        else:
            not_present_to_exclude_features = set(to_exclude_features) - set(X.columns)
            if len(not_present_to_exclude_features) > 0:
                raise warnings.warn(
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
        self._ebm.fit(X, y)
        self._one_hot_encoder.fit(X[self._categorical_features])

    # TODO : Provide function that give the quality of the binarizer via the learning metrics of ebm

    def transform(self, X):
        """Transform function of binarizer

        This function uses the previously fitted EBM to extract binary features from continuous features.
        For each continuous feature, it looks at each constructed interval to create a binary feature based on feature value belonging to this interval or not.
        For categorical features, it uses the one-hot encoder previously fitted.
        For features to exclude from the binarizer, it copies the values in the new dataset.

        Args:
            X (pandas.DataFrame): Dataframe of features to transform

        Returns:
            pandas.DataFrame: Binarized features
            pandas.DataFrame: DataFrame of information of binary feature and corresponding feature
        """

        if not (self._ebm.has_fitted_ and not check_is_fitted(self._one_hot_encoder)):
            raise NotFittedError("AutomaticFeatureBinarizer has not been fitted.")

        dict_check_continuous_features = {
            c: pa.Column(checks=[NumericCheck()])
            for c in self._continuous_features
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

        ebm_global = self._ebm.explain_global(name="EBM")

        X_binarized = pd.DataFrame()
        list_scores = []
        list_lower_threshold = []
        list_upper_threshold = []
        list_original_column = []

        # For each continuous feature, we look at their single-feature tree
        for i, feature_name in enumerate(ebm_global.data()["names"]):
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

                    # Case of the first plateau, only have an upper bound for current feature value
                    if j == 0:
                        col_name = f"{feature_name} < {np.round(threshold_upper,2)}"

                        X_binarized[col_name] = X[feature_name] < threshold_upper

                        list_lower_threshold.append(np.nan)
                        list_upper_threshold.append(threshold_upper)
                    # Case of the last plateau, only have a lower bound for current feature value
                    elif j == number_plateau - 1:
                        col_name = f"{feature_name} >= {np.round(threshold_lower,2)}"

                        X_binarized[col_name] = X[feature_name] >= threshold_lower
                        list_lower_threshold.append(threshold_lower)
                        list_upper_threshold.append(np.nan)

                    # Case for other inbetween plateaus
                    else:
                        col_name = f"{np.round(threshold_lower,2)} <= {feature_name} < {np.round(threshold_upper,2)}"

                        X_binarized[col_name] = (X[feature_name] < threshold_upper) & (
                            X[feature_name] >= threshold_lower
                        )

                        list_lower_threshold.append(threshold_lower)
                        list_upper_threshold.append(threshold_upper)

                    X_binarized[col_name] = X_binarized[col_name].astype(int)

                    list_scores.append(contrib)
                    list_original_column.append(feature_name)

        # TODO : Put logodds of ebm for categorical features

        # One-hot encode categorical features
        X_binarized[
            self._one_hot_encoder.get_feature_names_out()
        ] = self._one_hot_encoder.transform(X[self._categorical_features])

        # Add info for info dataframe
        for name_out in self._one_hot_encoder.get_feature_names_out():
            list_scores.append(None)
            list_lower_threshold.append(None)
            list_upper_threshold.append(None)
            list_original_column.append("_".join(name_out.split("_")[:-1]))

        # Copy features to exclude from binarizing
        X_binarized[self._to_exclude_features] = X[self._to_exclude_features].copy()
        for name_out in self._to_exclude_features:
            list_scores.append(None)
            list_lower_threshold.append(None)
            list_upper_threshold.append(None)
            list_original_column.append(name_out)

        # Regroup information about binary features created in the info dataframe
        # this dataframe contains:
        # - the list for each binary feature of log odds contribution
        # - the list for each binary feature of lower threshold used
        # - the list for each binary feature of upper threshold used
        # - the list for each binary feature of the original feature it came from

        df_score_feature = pd.DataFrame(
            index=X_binarized.columns,
            data=np.array(
                [
                    list_scores,
                    list_lower_threshold,
                    list_upper_threshold,
                    list_original_column,
                ]
            ).T,
            columns=[
                "EBM_log_odds_contribution",
                "lower_threshold",
                "upper_threshold",
                "feature",
            ],
        )

        # the intercept is the basis log odd value of the EBM without adding/substracting
        # the different log odds contribution of each feature
        row_intercept = pd.DataFrame(
            index=["intercept"],
            columns=df_score_feature.columns,
            data=[[self._ebm.intercept_[0], None, None, "intercept"]],
        )
        df_score_feature = pd.concat([df_score_feature, row_intercept], axis=0)
        df_score_feature.index.name = "binary_feature"
        return X_binarized, df_score_feature
