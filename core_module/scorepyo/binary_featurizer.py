"""TODO Add docstring binary featurizer
TODO Add typing
TODO Comment code
TODO Improve naming and code clarity
TODO eneble possibility of regressor?

Returns:

    _type_: _description_
"""
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.preprocessing import OneHotEncoder


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


    Methods
    -------
    fit(X_train, y_train, categorical_features="auto", binary_features=None)
        fits the EBM model on X,y and the one-hot encoder on the categorical features

    transform(X, categorical_features="auto", binary_features="auto")
        transforms features in X into binarized features based on previously fitted EBM and one-hot encoder
    """

    def __init__(
        self,
        max_number_binaries_by_features=2,
        keep_negative=False,
    ):
        """
        Args:
            max_number_binaries_by_features (int, optional): maximum number of binary features to compute by feature. Defaults to 2.
            keep_negative (bool, optional): indicator to keep features that decrease predicted probability. Defaults to False.
        """
        # TODO raise exception
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

    def fit(self, X, y, categorical_features="auto"):
        """Fit function of binarizer

        This functions fits the EBM on X,y and the one-hot encoder on X.

        Args:
            X (pandas.Dataframe): Dataframe of features to use to fit the binarizer
            y (pandas.Series): target values
            categorical_features (list(str), optional): list of categorical features to one-hot encode.
            Defaults to "auto" to automatically detect categorical features.
        """

        # TODO Use pandera
        if categorical_features == "auto":
            categorical_features = None
        self._ebm.fit(X, y)

        self._one_hot_encoder.fit(X[self.categorical_features])
        self._categorical_features = categorical_features

    def transform(self, X, categorical_features="auto"):
        """Transform function of binarizer

        This function uses the previously fitted EBM to extract binary features from continuous features.
        For each feature, it looks at each constructed interval to create a binary feature based on feature value belonging to this interval or not.

        Args:
            X (pandas.DataFrame): Dataframe of features to transform
            categorical_features list(str): list of categorical. Defaults to "auto".

        Returns:
            pandas.DataFrame: Binarized features
            pandas.DataFrame: DataFrame of information of binary feature and corresponding feature
        """

        # TODO Use pandera
        ebm_global = self._ebm.explain_global(name="EBM")

        X_binarized = pd.DataFrame()
        list_scores = []
        list_lower_threshold = []
        list_upper_threshold = []
        list_original_column = []

        if categorical_features == "auto":
            # TODO do something about categorical data
            categorical_features = []

        for i, feature_name in enumerate(ebm_global.data()["names"]):
            if feature_name not in categorical_features:

                dico_feature_i = ebm_global.data(i)

                number_plateau = len(ebm_global.data(i)["scores"])

                for j in range(number_plateau):
                    contrib = dico_feature_i["scores"][j]

                    if (contrib < 0) and (not self.keep_negative):
                        continue
                    threshold_lower = dico_feature_i["names"][j]
                    threshold_upper = dico_feature_i["names"][j + 1]

                    feature_name_binarized_lower = (
                        f"{feature_name} >= {np.round(threshold_lower,2)}"
                    )
                    feature_name_binarized_upper = (
                        f"{feature_name} < {np.round(threshold_upper,2)}"
                    )

                    feature_name_binarized_inbetween = f"{np.round(threshold_lower,2)} <= {feature_name} < {np.round(threshold_upper,2)}"

                    if j == 0:
                        col_name = feature_name_binarized_upper

                        X_binarized[col_name] = X[feature_name] < threshold_upper

                        list_lower_threshold.append(np.nan)
                        list_upper_threshold.append(threshold_upper)
                    elif j == number_plateau - 1:
                        col_name = feature_name_binarized_lower

                        X_binarized[col_name] = X[feature_name] >= threshold_lower
                        list_lower_threshold.append(threshold_lower)
                        list_upper_threshold.append(np.nan)
                    else:
                        col_name = feature_name_binarized_inbetween

                        X_binarized[col_name] = (X[feature_name] < threshold_upper) & (
                            X[feature_name] >= threshold_lower
                        )

                        list_lower_threshold.append(threshold_lower)
                        list_upper_threshold.append(threshold_upper)

                    X_binarized[col_name] = X_binarized[col_name].astype(int)

                    list_scores.append(contrib)
                    list_original_column.append(feature_name)

        # TODO : Put logodds of ebm ofr categorical features
        if self.one_hot_encode is not None:

            X_binarized[
                self._one_hot_encoder.get_feature_names_out()
            ] = self._one_hot_encoder.transform(X[categorical_features])

            for name_out in self._one_hot_encoder.get_feature_names_out():
                list_scores.append(None)
                list_lower_threshold.append(None)
                list_upper_threshold.append(None)
                list_original_column.append("_".join(name_out.split("_")[:-1]))
        else:
            X_binarized[categorical_features] = X[categorical_features]
            for name_out in categorical_features:
                list_scores.append(None)
                list_lower_threshold.append(None)
                list_upper_threshold.append(None)
                list_original_column.append(name_out)

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

        row_intercept = pd.DataFrame(
            index=["intercept"],
            columns=df_score_feature.columns,
            data=[[self._ebm.intercept_[0], None, None, "intercept"]],
        )
        df_score_feature = pd.concat([df_score_feature, row_intercept], axis=0)
        df_score_feature.index.name = "binary_feature"
        return X_binarized, df_score_feature
