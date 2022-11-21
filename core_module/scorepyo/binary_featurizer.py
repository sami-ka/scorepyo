import numpy as np
import pandas as pd


from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.preprocessing import OneHotEncoder


class AutomaticBinaryFeaturizer:
    def __init__(
        self,
        max_number_binaries_by_features=2,
        one_hot_encode=True,
        keep_negative=False,
    ):
        # TODO raise exception
        self.max_number_binaries_by_features = max_number_binaries_by_features
        self.one_hot_encode = one_hot_encode

        self.keep_negative = keep_negative

        self._ebm = ExplainableBoostingClassifier(
            interactions=0, max_bins=self.max_number_binaries_by_features + 2
        )

        self.one_hot_encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", max_categories=10, sparse=False
        )

    def fit(self, X_train, y_train, categorical_features="auto"):
        if categorical_features == "auto":
            categorical_features = None
        self._ebm.fit(X_train, y_train)

        self.one_hot_encoder.fit(X_train[categorical_features])

    def transform(self, X, X_train, X_test, categorical_features="auto"):
        ebm_global = self._ebm.explain_global(name="EBM")

        # X_train_binarized = pd.DataFrame()
        # X_test_binarized = pd.DataFrame()
        X_binarized = pd.DataFrame()
        list_scores = []
        list_lower_threshold = []
        list_upper_threshold = []
        list_original_column = []

        if categorical_features == "auto":
            categorical_features = None

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
                    if j == 0:
                        col_name = feature_name_binarized_upper
                        # X_train_binarized[col_name] = (
                        #     X_train[feature_name] < threshold_upper
                        # )
                        # X_test_binarized[col_name] = (
                        #     X_test[feature_name] < threshold_upper
                        # )
                        X_binarized[col_name] = X[feature_name] < threshold_upper

                        list_lower_threshold.append(np.nan)
                        list_upper_threshold.append(threshold_upper)
                    elif j == number_plateau - 1:
                        col_name = feature_name_binarized_lower
                        # X_train_binarized[col_name] = (
                        #     X_train[feature_name] >= threshold_lower
                        # )
                        # X_test_binarized[col_name] = (
                        #     X_test[feature_name] >= threshold_lower
                        # )
                        X_binarized[col_name] = X[feature_name] >= threshold_lower
                        list_lower_threshold.append(threshold_lower)
                        list_upper_threshold.append(np.nan)
                    else:
                        col_name = (
                            feature_name_binarized_lower
                            + " and "
                            + feature_name_binarized_upper
                        )
                        # X_train_binarized[col_name] = (
                        #     X_train[feature_name] < threshold_upper
                        # ) & (X_train[feature_name] >= threshold_lower)
                        # X_test_binarized[col_name] = (
                        #     X_test[feature_name] < threshold_upper
                        # ) & (X_test[feature_name] >= threshold_lower)
                        X_binarized[col_name] = (X[feature_name] < threshold_upper) & (
                            X[feature_name] >= threshold_lower
                        )

                        list_lower_threshold.append(threshold_lower)
                        list_upper_threshold.append(threshold_upper)

                    X[col_name] = X[col_name].astype(int)

                    list_scores.append(contrib)
                    list_original_column.append(feature_name)

        if self.one_hot_encode is not None:

            X[
                self.one_hot_encoder.get_feature_names_out()
            ] = self.one_hot_encoder.transform(X[categorical_features])
            # X_test_binarized[

            #     one_hot_encoder.get_feature_names_out()
            # ] = one_hot_encoder.transform(X_test[categorical_features])

            for name_out in self.one_hot_encoder.get_feature_names_out():
                list_scores.append(None)
                list_lower_threshold.append(None)
                list_upper_threshold.append(None)
                list_original_column.append("_".join(name_out.split("_")[:-1]))
        else:
            # X[categorical_features] = self.one_hot_encoder.transform(
            #     X[categorical_features]
            # )
            for name_out in categorical_features:
                list_scores.append(None)
                list_lower_threshold.append(None)
                list_upper_threshold.append(None)
                list_original_column.append(name_out)

        df_score_feature = pd.DataFrame(
            index=X_train_binarized.columns,
            data=np.array(
                [
                    list_scores,
                    list_lower_threshold,
                    list_upper_threshold,
                    list_original_column,
                ]
            ).T,
            columns=[
                "contrib",
                "lower_threshold",
                "upper_threshold",
                "original_feature",
            ],
        )

        row_intercept = pd.DataFrame(
            index=["intercept"],
            columns=df_score_feature.columns,
            data=[[ebm.intercept_[0], None, None, "intercept"]],
        )
        df_score_feature = pd.concat([df_score_feature, row_intercept], axis=0)

        return X_train_binarized, X_test_binarized, df_score_feature
