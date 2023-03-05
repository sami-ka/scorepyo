"""Classes to create and fit risk-score type model.
"""
import itertools
import numbers
import time
from abc import abstractmethod
from typing import Optional

import cvxpy as cp
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pandera as pa
from sklearn.exceptions import NotFittedError

from scorepyo._utils import fast_numba_auc
from scorepyo.binarizers import EBMBinarizer
from scorepyo.exceptions import (
    MinPointOverMaxPointError,
    NegativeValueError,
    NonIntegerValueError,
)


class _BaseRiskScore:
    """
    Base class for risk score type model.

    This class provides common functions for risk score type model, no matter the way points and binary features are designed.
    It needs common attributes, such as minimum/maximum point value for each binary feature and number of selected binary feature.


    Attributes
    ----------
    nb_max_features : int
        maximum number of binary features to compute by feature
    min_point_value : int
        minimum points to assign for a binary feature
    max_point_value: ExplainableBoostingClassifier
        maximum points to assign for a binary feature
    _df_info: pandas.DataFrame
        Dataframe containing the link between a binary feature and its origin



    Methods
    -------
    @staticmethod
    _predict_proba_score(score, intercept):
        computes the value of the logistic function at score+intercept value

    @abstractmethod
    fit(self, X, y):
        function to be implemented by child classes. This function should create the feature-point card and score card attributes.

    predict(self, X, proba_threshold=0.5):
        function that predicts the positive/negative class by using a threshold on the probability given by the model

    predict_proba(self, X):
        function that predicts the probability of the positive class

    summary(self):
        function that prints the feature-point card and score card of the model
    """

    _DESCRIPTION_COL = "Description"
    """Column name for binary feature in the risk score summary"""

    _POINT_COL = "Point(s)"
    """Column name for points in the risk score summary"""

    _FEATURE_COL = "Feature"
    """Column name for original feature in the risk score summary"""

    def __init__(
        self,
        nb_max_features: int = 4,
        min_point_value: int = -2,
        max_point_value: int = 3,
        df_info: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            nb_max_features (int): Number of maximum binary features to be selected
            min_point_value (int, optional): Minimum point assigned to a binary feature. Defaults to -2.
            max_point_value (int, optional): Maximum point assigned to a binary feature. Defaults to 3.
            df_info (pandas.DataFrame, optional): DataFrame linking original feature and binary feature. Defaults to None.
        """
        if nb_max_features <= 0:
            raise NegativeValueError(
                f"nb_max_features must be a strictly positive integer. \n {nb_max_features} is not positive."
            )
        if not isinstance(nb_max_features, numbers.Integral):
            raise NonIntegerValueError(
                f"nb_max_features must be a strictly positive integer. \n {nb_max_features} is not an integer."
            )
        self.nb_max_features: int = nb_max_features

        if not isinstance(min_point_value, numbers.Integral):
            raise NonIntegerValueError(
                f"min_point_value must be an integer. \n {min_point_value} is not an integer."
            )

        if not isinstance(max_point_value, numbers.Integral):
            raise NonIntegerValueError(
                f"max_point_value must be an integer. \n {max_point_value} is not an integer."
            )
        if min_point_value > max_point_value:
            raise MinPointOverMaxPointError(
                f"max_point_value must be greater than or equal to min_point_value. \n {max_point_value=} < {min_point_value=} ."
            )
        self.min_point_value: int = min_point_value
        self.max_point_value: int = max_point_value

        self._df_info: Optional[pd.DataFrame]

        if df_info is not None:
            dataframe_schema = pa.DataFrameSchema(
                {
                    "binary_feature": pa.Column(),
                    "feature": pa.Column(),
                },
                strict=False,  # Disable check of other columns in the dataframe
            )

            dataframe_schema.validate(df_info)
            self._df_info = df_info.copy()
        else:
            self._df_info = None

        self.feature_point_card: Optional[pd.DataFrame] = None
        self.score_card: Optional[pd.DataFrame] = None

    @staticmethod
    def _predict_proba_score(
        score: int, intercept: float, multiplier: Optional[float] = 1.0
    ) -> np.ndarray:
        """Function that computes the logistic function value at score+intercept value

        Args:
            score (np.array(int)): sum of points coming from binary features
            intercept (float): intercept of score card. log-odds of having 0 point

        Returns:
            np.array(int): associated probability
        """
        score_associated_proba = 1 / (1 + np.exp(-(score + intercept) / multiplier))
        return score_associated_proba

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Optional[NotImplementedError]:
        """Functions that creates the feature-point card and score card

        Must be defined for each child class

        Args:
            X (pandas.DataFrame): binary feature dataset
            y (pandas.Series): binary target

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, proba_threshold: float = 0.5) -> np.ndarray:
        """Predicts binary class based on probabillity threshold

        Afer computing the probability for each sample,

        Args:
            X (pd.DataFrame): _description_
            proba_threshold (float, optional): probability threshold for binary classification. Defaults to 0.5.

        Returns:
            nbarray of shape (n_samples,): predicted class based on predicted probability and threshold
        """
        if (self.feature_point_card is None) or (self.score_card is None):
            raise NotFittedError("RiskScore model has not been fitted yet")
        proba = self.predict_proba(X)

        return (proba[:, 1] >= proba_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Function that outputs probability of positive class according to risk-score model

        Args:
            X (pandas.DataFrame): dataset of features

        Returns:
            ndarray of shape (n_samples, 2): probability of negative and positive class in each column resp.
        """
        if (self.feature_point_card is None) or (self.score_card is None):
            raise NotFittedError("RiskScore model has not been fitted yet")

        # TODO have the same behaviour as predict proba from sklearn
        # TODO check if full numpy approach is faster
        df_score_card = self.score_card.copy().T.set_index("SCORE")
        list_features = self.feature_point_card["binary_feature"].values
        # TODO check that X only has selected featureso r just that it contains them?

        dataframe_schema = pa.DataFrameSchema(
            {
                c: pa.Column(checks=[pa.Check.isin([0, 1, 0.0, 1.0])])
                for c in list_features
            },
            strict=False,  # Disable check of other columns in the dataframe
        )

        dataframe_schema.validate(X)

        X_selected_features = X[list_features]

        points = self.feature_point_card[self._POINT_COL].values
        X_total_points = np.matmul(X_selected_features.values, points)
        proba = df_score_card.loc[X_total_points, "_RISK_FLOAT"].values.reshape(-1, 1)

        return np.concatenate([1 - proba, proba], axis=1)

    def summary(self) -> None:
        if (self.feature_point_card is None) or (self.score_card is None):
            raise NotFittedError("RiskScore model has not been fitted yet")
        feature_point_summary = (
            self.feature_point_card[[self._DESCRIPTION_COL, self._POINT_COL]]
            .sort_values(by=self._POINT_COL)
            .copy()
        )

        empty_column_name = " "
        feature_point_summary[empty_column_name] = "..."
        feature_point_summary.iloc[1:, -1] = "+ " + feature_point_summary.iloc[1:, -1]
        feature_point_summary.iloc[0, -1] = " " + feature_point_summary.iloc[0, -1]
        additional_display_rows = pd.DataFrame(
            data=[
                # [None, None, None],
                [" ", "SCORE=", " ..."]
            ],
            columns=feature_point_summary.columns,
            index=[" "],
        )

        feature_point_summary = pd.concat(
            [feature_point_summary, additional_display_rows], axis=0, ignore_index=False
        )

        feature_point_summary.index.name = self._FEATURE_COL

        print("======================")
        print("| FEATURE-POINT CARD |")
        print("======================")
        print(feature_point_summary.to_markdown())
        print()
        print("=======================================")
        print("=======================================")
        print()
        print("======================")
        print("|     SCORE CARD     |")
        print("======================")
        print(self.score_card.loc[["SCORE", "RISK"], :].to_markdown(headers="firstrow"))


class RiskScore(_BaseRiskScore):
    """
    Risk score model based on a binarizer, ranking of features and exhaustive enumeration.

    This class is a child class of _BaseRiskScore. It implements the fit method that creates the feature-point card and score card attribute.
    It computes them by binarizing features based on EBM, ranking binary features and doing an exhaustive enumeration of features combination.
    It performs a hierarchical optimization:
    1) first it optimizes ROC AUC and/or PR AUC by selecting points for all combinations of binary feature;
    2) once the binary feature combination is chosen with the corresponding interger points, the logloss is optimized for each possible sum of points.

    Attributes
    ----------
    max_number_binaries_by_features : int
        maximum number of binary features to compute by feature
    nb_max_features: int
        maximum number of binary features to select
    min_point_value : int
        minimum points to assign for a binary feature
    max_point_value: ExplainableBoostingClassifier
        maximum points to assign for a binary feature
    _df_info: pandas.DataFrame
        Dataframe containing the link between a binary feature and its origin



    Methods
    -------
    fit(self, X, y):
        function creating the feature-point card and score card attributes via Optuna



    From _BaseRiskScore:

    @staticmethod
    _predict_proba_score(score, intercept):
        computes the value of the logistic function at score+intercept value

    predict(self, X, proba_threshold=0.5):
        function that predicts the positive/negative class by using a threshold on the probability given by the model

    predict_proba(self, X):
        function that predicts the probability of the positive class

    summary(self):
        function that prints the feature-point card and score card of the model
    """

    def __init__(
        self,
        binarizer,
        nb_max_features: int = 4,
        min_point_value: int = -2,
        max_point_value: int = 3,
        max_number_binaries_by_features: int = 3,
        nb_additional_features: Optional[int] = 4,
        df_info: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            nb_max_features (int): Number of maximum binary features to be selected
            min_point_value (int, optional): Minimum point assigned to a binary feature. Defaults to -2.
            max_point_value (int, optional): Maximum point assigned to a binary feature. Defaults to 3.
            max_number_binaries_by_features (int, optional): Maximum number of binary features by original feature. Defaults to 3.
            nb_additional_features (int, optional): Number of additional features to consider for the binary feature selection. Defaults to 3.
            df_info (pandas.DataFrame, optional): DataFrame linking original feature and binary feature. Defaults to None.
        """
        super().__init__(nb_max_features, min_point_value, max_point_value, df_info)

        # TODO Check value
        self.max_number_binaries_by_features: int = max_number_binaries_by_features
        self.binarizer = binarizer
        self.nb_additional_features = int(nb_additional_features)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ranker,
        calibrator,
        X_calib: pd.DataFrame = None,
        y_calib: pd.Series = None,
        categorical_features=None,
        fit_binarizer=True,
        enumeration_maximization_metric=fast_numba_auc,
    ):
        """Function that search best parameters (choice of binary features, points and probabilities) of a risk score model.


        It computes them by binarizing features based on EBM, ranking binary features and doing an exhaustive enumeration of features combination.
        It performs a hierarchical optimization:
        1) first it optimizes ROC AUC and/or PR AUC by selecting points for all combinations of selected binary feature.
        The selection of binary features combination is done by:
            a) ranking the binary features,
            b) taking the top features according to the ranking,
            c) enumerate all combinations of binary features,
            d) enumerate all point assignment for each combination of binary features generated.
            e) choose the best combination of binary feature and point based on a ranking metric
        Different ranking techniques are available for step b) :
        LogOddsDensity, DiverseLogOddsDensity, CumulativeMetric, BordaRank, LassoPathRank, LarsPathRank, OMPRank, FasterRiskRank


        2) once the binary feature combination is chosen with the corresponding interger points, the logloss is optimized for each possible sum of points.
        The logloss optimization can be done on a different dataset (X_calib, y_calib).
        It can be done with a vanilla mode (preferred when train or calibration set is big enough), or a bootstrapped mode to avoid overfitting when there are few samples.
        These logloss optimizer with more details can be found in the calibration script of the package.

        Args:
            X (pandas.DataFrame): Dataset of features to fit the risk score model on
            y (pandas.Series): Target binary values
            X_calib (pandas.DataFrame): Dataset of features to calibrate probabilities on
            y_calib (pandas.Series): Target binary values for calibration
            categorical_features: list of categorical features for the binarizer
            fit_binarizer: boolean to indicate the binarizer should be fitted or not
            sorting_method: ranking method of features #TODO change to ranker object
            optimization_method: optimization method to define probabilities (logloss)
        """
        start_time = time.time()

        # Binarize the features with the AutoBinarizer class
        if fit_binarizer:
            self.binarizer.fit(X, y, categorical_features=categorical_features)
        df_info = self.binarizer.df_score_feature

        start_time_P1 = time.time()

        # Prepare calibration set
        if X_calib is None:
            X_calib = X.copy()
            y_calib = y.copy()

        # Binarize the features with the previously fitted binarizer
        X_binarized = self.binarizer.transform(X)
        X_calib_binarized = self.binarizer.transform(X_calib)

        fit_transform_time = time.time() - start_time

        # Rank the binary feature by likeliness to be important for the risk score model

        self.binarizer.old_df_info = df_info.copy()

        df_ranker = df_info[["log_odds", "density", "feature"]].copy()

        df_rank = ranker.compute_ranking_features(
            df=df_ranker,
            X_binarized=X_binarized,
            y=y,
            nb_steps=(self.nb_additional_features + self.nb_max_features),
        )

        df_info = df_info.merge(
            df_rank, right_index=True, left_index=True, how="left", validate="1:1"
        )
        df_info = df_info.sort_values(by="rank", ascending=True)

        pool_top_features = df_info.index[
            : self.nb_max_features + self.nb_additional_features
        ]

        # Compute bounds for points for each feature to reduce optimization space
        # -> negative points for negative log odds
        # -> positive points for positive log odds
        df_sense = df_info.iloc[
            : self.nb_max_features + self.nb_additional_features
        ].copy()

        df_sense.loc[:, "lower_bound_point"] = np.where(
            df_sense["log_odds"] > 0,
            min([1, self.max_point_value]),
            self.min_point_value,
        )

        df_sense.loc[:, "upper_bound_point"] = np.where(
            df_sense["log_odds"] > 0,
            self.max_point_value,
            max([-1, self.min_point_value]),
        )

        # Define all possible integer values for each binary feature
        dict_point_ranges = {
            f: {
                "all_points": np.arange(
                    df_sense.loc[f, "lower_bound_point"],
                    df_sense.loc[f, "upper_bound_point"] + 1e-3,
                    1,
                ),
            }
            for f in pool_top_features
        }

        best_metric = 1e9

        _count = 0
        # For all combinations of nb_max_feature from the set of selected binary features
        nb_combi = len(
            list(itertools.combinations(pool_top_features, self.nb_max_features))
        )

        # Max number of point combination
        dim_max_point_combination = (
            max(len(dict_point_ranges[f]["all_points"]) for f in pool_top_features)
            ** self.nb_max_features
        )

        # Compute cube of all points combination :
        # number of combinations of self.nb_max_features feature among the selected pool of binary features x cardinality of feature pool
        cube = np.zeros(
            shape=(nb_combi * dim_max_point_combination, len(pool_top_features)),
            dtype=np.float16,
        )

        end_time_P1 = time.time()

        start_time_P2 = time.time()

        idx = 0
        # for each feature combination, compute point combinations
        for i, top_features in enumerate(
            itertools.combinations(pool_top_features, self.nb_max_features)
        ):
            # Gather all point ranges for each binary feature
            # if the feature is not in the current selected combination, the only point possibility is 0
            all_points_by_feature = [
                dict_point_ranges[f]["all_points"] if f in top_features else [0]
                for f in pool_top_features
            ]

            # Compute the cartesian product of all possible point values for each feature
            # This creates a nxd matrix with n being the number of combinations of points for each binary feature
            # and d being the total number of binary features in pool_top_features.
            # for each line, there are only self.nb_max_features non zero point values.
            all_points_possibilities = np.array(
                list(itertools.product(*all_points_by_feature))
            )

            # Put this points combination into the global enumeration of points
            cube[
                idx : idx + all_points_possibilities.shape[0], :
            ] = all_points_possibilities
            idx += all_points_possibilities.shape[0]

        end_time_P2 = time.time()

        # if chunk_size_cube is None:
        chunk_size_cube = (50, len(pool_top_features), 20)

        start_time_P4 = time.time()

        # Use dask to compute the metric to optimize on all points combination

        # Dask array for the global enumeration of combinations
        dask_cube = da.from_array(cube, chunks=((50, len(pool_top_features))))

        # Dask array for the dataset of selected binary features
        dataset_transpose = X_binarized[list(pool_top_features)].values.T
        dask_dataset_T = da.from_array(
            dataset_transpose, chunks=dataset_transpose.shape
        )

        # 1Dfication of maximization metric function

        enumeration_optimization_metric_1d = lambda y_score: [
            enumeration_maximization_metric(y.values, y_score)
        ]

        # computation of score for all samples for all enumerated points combination
        matmul = da.matmul(dask_cube, dask_dataset_T)

        # computation of the maximum value on the defined enumeration metric
        idx_max = da.argmax(
            da.apply_along_axis(
                enumeration_optimization_metric_1d,
                axis=1,
                arr=matmul,
                shape=(1,),
                dtype=matmul.dtype,
            )
        ).compute()

        # Getting the points combination with the maximumu value
        best_points = cube[idx_max, :]
        best_feature_and_point_selection = [
            (point, f)
            for point, f in zip(best_points, list(pool_top_features))
            if point != 0
        ]

        best_scenario, best_feature_selection = zip(*best_feature_and_point_selection)
        best_feature_selection = list(best_feature_selection)

        end_time_P4 = time.time()

        # Probabilities computation
        start_time_P5 = time.time()

        df_calibration = pd.DataFrame(columns=["target", "score"])
        df_calibration["target"] = y_calib.values
        df_calibration["score"] = np.matmul(
            X_calib_binarized[best_feature_selection].values, best_scenario
        )

        min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
        max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))

        calibrator.calibrate(df_calibration, min_sum_point, max_sum_point)

        df_reordering = calibrator.calibrate(
            df_calibration, min_sum_point, max_sum_point
        )

        end_time_P5 = time.time()

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=best_feature_selection)
        self.feature_point_card[self._POINT_COL] = best_scenario

        self.feature_point_card["binary_feature"] = self.feature_point_card.index.values
        self.feature_point_card[
            self._FEATURE_COL
        ] = self.feature_point_card.index.values

        self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
        self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
            "binary_feature"
        ].values

        possible_scores = list(df_reordering.index)
        possible_risks = df_reordering["sorted_proba"]
        # Compute nice display of probabilities
        possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

        # Assign dataframe to score_card attribute
        self.score_card = pd.DataFrame(
            index=["SCORE", "RISK", "_RISK_FLOAT"],
            data=[possible_scores, possible_risks_pct, possible_risks],
        )
        self._total_fit_time = time.time() - start_time
        self._model_fit_time = self._total_fit_time - fit_transform_time

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_binarized = self.binarizer.transform(X)
        return super().predict_proba(X_binarized)

    def _fit_numpy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_calib: pd.DataFrame = None,
        y_calib: pd.Series = None,
        categorical_features=None,
    ):
        start_time = time.time()
        # Binarize the features with the AutoBinarizer class
        self.binarizer.fit(X, y, categorical_features=categorical_features)
        df_info = self.binarizer.df_score_feature

        if X_calib is None:
            X_calib = X.copy()
            y_calib = y.copy()
        X_binarized = self.binarizer.transform(X)
        X_calib_binarized = self.binarizer.transform(X_calib)

        # Rank the binary feature by likeliness to be important for the risk score model
        # The current estimated importance is the log odd computed by the EBM model x number of positive samples for
        # that binary feature.
        # The cardinality of the positive samples is de emphasize by taking the 0.95 power.
        # TODO: take into account the impact of having mixed class in the samples

        df_info["abs_contribution"] = df_info["log_odds"].abs() * df_info[
            "density"
        ].fillna(0).astype(int).pow(
            0.95
        )  # to de emphasize large values impact

        # Compute the reduced pool of top features to choose from
        pool_top_features = df_info.sort_values(
            by="abs_contribution", ascending=False
        ).index[: self.nb_max_features + self.nb_additional_features]

        # Compute bounds for points for each feature to reduce optimization space
        # -> negative points for negative log odds
        # -> positive points for positive log odds
        df_sense = df_info.sort_values(by="abs_contribution", ascending=False).iloc[
            : self.nb_max_features + self.nb_additional_features
        ]
        df_sense["lower_bound_point"] = np.where(
            df_sense["log_odds"] > 0,
            min([1, self.max_point_value]),
            self.min_point_value,
        )
        df_sense["upper_bound_point"] = np.where(
            df_sense["log_odds"] > 0,
            self.max_point_value,
            max([-1, self.min_point_value]),
        )

        # Define all possible integer values for each binary feature
        dict_point_ranges = {
            f: {
                "all_points": np.arange(
                    df_sense.loc[f, "lower_bound_point"],
                    df_sense.loc[f, "upper_bound_point"] + 1e-3,
                    1,
                ),
            }
            for f in pool_top_features
        }

        best_metric = 1e9

        _count = 0
        # For all combinations of nb_max_feature from the set of selected binary features
        nb_combi = len(
            list(itertools.combinations(pool_top_features, self.nb_max_features))
        )

        # Max number of point combination
        dim_max_point_combination = (
            max(len(dict_point_ranges[f]["all_points"]) for f in pool_top_features)
            ** self.nb_max_features
        )

        # Compute cube of all points combination :
        # number of k feature among n x cardinality of feature pool x max number of point combination for k features
        cube = np.zeros(
            shape=(nb_combi, len(pool_top_features), dim_max_point_combination),
            dtype=np.float16,
        )

        # for each feature combination, compute point combinations
        for i, top_features in enumerate(
            itertools.combinations(pool_top_features, self.nb_max_features)
        ):
            # Gather all point ranges for each binary feature
            all_points_by_feature = [
                dict_point_ranges[f]["all_points"] if f in top_features else [0]
                for f in pool_top_features
            ]

            # Compute the cartesian product of all possible point values for each feature
            # This creates a nxd matrix with n being the number of combinations of points for each binary feature
            # and d being the number of selected binary features
            all_points_possibilities = np.array(
                list(itertools.product(*all_points_by_feature))
            )
            all_points_possibilities = all_points_possibilities.T
            #     all_points_possibilities = all_points_possibilities.reshape(len(pool_top_features),-1)
            cube[i, :, : all_points_possibilities.shape[-1]] = all_points_possibilities

        # TODO uncomment for the non dask version
        cube_augmented = np.einsum(
            "ijk,jl->ijkl",
            cube,
            X_binarized[list(pool_top_features)].values.T,
            optimize="optimal",
            dtype=np.int8,
            casting="unsafe",
        )

        # score for each feature combi, for each point possibilities, for each sample
        score_all_case = cube_augmented.sum(axis=1)

        auc_results = np.zeros(shape=(score_all_case.shape[0], score_all_case.shape[1]))
        for i in range(score_all_case.shape[0]):
            for j in range(score_all_case.shape[1]):
                auc_results[i, j] = fast_numba_auc(
                    y.values, y_score=score_all_case[i, j, :]
                )

        flatten_max_index = auc_results.argmax()
        idx_max = np.unravel_index(flatten_max_index, auc_results.shape)
        best_metric = auc_results[idx_max[0], idx_max[1]]
        best_points = cube[idx_max[0], :, idx_max[1]]
        best_feature_and_point_selection = [
            (point, f)
            for point, f in zip(best_points, list(pool_top_features))
            if point != 0
        ]
        best_scenario, best_feature_selection = zip(*best_feature_and_point_selection)
        best_feature_selection = list(best_feature_selection)

        df_calibration = pd.DataFrame(columns=["target", "score"])
        df_calibration["target"] = y_calib.values
        df_calibration["score"] = np.matmul(
            X_calib_binarized[best_feature_selection].values, best_scenario
        )

        df_score_proba_association = df_calibration.groupby("score")["target"].mean()
        df_score_proba_association.columns = ["proba"]
        min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
        max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))
        full_index = np.arange(min_sum_point, max_sum_point + 1e-3)
        missing_index = set(full_index) - set(df_score_proba_association.index)

        df_score_proba_association = df_score_proba_association.reindex(full_index)
        df_score_proba_association = df_score_proba_association.interpolate(
            method="linear"
        )

        df_reordering = pd.DataFrame(df_score_proba_association.copy())
        # Count cardinality of each score
        df_reordering["count"] = df_calibration.groupby("score")["target"].count()
        df_reordering["count"] = df_reordering["count"].fillna(0)
        df_reordering["target"] = df_reordering["target"].fillna(0)

        # Adjust probability to have an optimized logloss and calibration
        df_cvx = df_reordering.copy()

        # Compute number of positive and negative samples at each score value
        positive_sample_count = (
            df_cvx["target"].values * df_cvx["count"].values
        ).astype(int)
        negative_sample_count = df_cvx["count"].values - positive_sample_count

        # Declare the list of probabilities to be set as variables
        list_proba = [cp.Variable(1) for _ in df_cvx.index]

        # Compute total size of samples to normalize the logloss
        total_count = df_cvx["count"].sum()

        # Compute the logloss at each score in a list in order to sum it later
        # the logloss at each score is simple as all samples will have the same
        # probability p. for all positive samples, add -log(p), for all negative samples add -log(1-p)
        list_expression = [
            -cp.log(p) * w_pos - cp.log(1 - p) * w_neg
            for p, w_pos, w_neg in zip(
                list_proba, positive_sample_count, negative_sample_count
            )
        ]
        objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objective

        # Declare the constraints for the probabilities
        # the probability at each score should be higher than probabilities at a lower score
        constraints = []
        for p in list_proba:
            constraints.append(p >= 0)
            constraints.append(p <= 1)
        for i in range(1, len(list_proba)):
            # TODO : Put the threshold away and combine all similar scores into 1
            constraints.append(list_proba[i] - list_proba[i - 1] - 1e-3 >= 0)

        problem = cp.Problem(objective, constraints)

        opt = problem.solve(verbose=False)

        # Get the optimized value for the probabilities
        df_reordering["sorted_proba"] = [p.value[0] for p in list_proba]

        self.df_reordering_debug = df_reordering

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=best_feature_selection)
        self.feature_point_card[self._POINT_COL] = best_scenario

        self.feature_point_card["binary_feature"] = self.feature_point_card.index.values
        self.feature_point_card[
            self._FEATURE_COL
        ] = self.feature_point_card.index.values

        self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
        self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
            "binary_feature"
        ].values

        possible_scores = list(df_reordering.index)
        possible_risks = df_reordering["sorted_proba"]
        # Compute nice display of probabilities
        possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

        # Assign dataframe to score_card attribute
        self.score_card = pd.DataFrame(
            index=["SCORE", "RISK", "_RISK_FLOAT"],
            data=[possible_scores, possible_risks_pct, possible_risks],
        )
        self._total_fit_time = time.time() - start_time
        self._model_fit_time = self._total_fit_time - self.binarizer._fit_time

    def _fit_dask_old(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_calib: pd.DataFrame = None,
        y_calib: pd.Series = None,
        categorical_features=None,
        chunk_size_cube=None,
        chunk_size_data=None,
        fit_binarizer=True,
    ):
        start_time = time.time()

        # Binarize the features with the AutoBinarizer class
        if fit_binarizer:
            self.binarizer.fit(X, y, categorical_features=categorical_features)
        df_info = self.binarizer.df_score_feature

        start_time_P1 = time.time()
        if X_calib is None:
            X_calib = X.copy()
            y_calib = y.copy()
        X_binarized = self.binarizer.transform(X)
        X_calib_binarized = self.binarizer.transform(X_calib)

        fit_transform_time = time.time() - start_time
        # Rank the binary feature by likeliness to be important for the risk score model
        # The current estimated importance is the log odd computed by the EBM model x number of positive samples for
        # that binary feature.
        # The cardinality of the positive samples is de emphasize by taking the 0.95 power.
        # TODO: take into account the impact of having mixed class in the samples

        df_info["abs_contribution"] = df_info["log_odds"].abs() * df_info[
            "density"
        ].fillna(0).astype(int).pow(
            0.95
        )  # to de emphasize large values impact

        # Compute the reduced pool of top features to choose from
        pool_top_features = df_info.sort_values(
            by="abs_contribution", ascending=False
        ).index[: self.nb_max_features + self.nb_additional_features]

        # Compute bounds for points for each feature to reduce optimization space
        # -> negative points for negative log odds
        # -> positive points for positive log odds
        df_sense = df_info.sort_values(by="abs_contribution", ascending=False).iloc[
            : self.nb_max_features + self.nb_additional_features
        ]
        df_sense["lower_bound_point"] = np.where(
            df_sense["log_odds"] > 0,
            min([1, self.max_point_value]),
            self.min_point_value,
        )
        df_sense["upper_bound_point"] = np.where(
            df_sense["log_odds"] > 0,
            self.max_point_value,
            max([-1, self.min_point_value]),
        )

        # Define all possible integer values for each binary feature
        dict_point_ranges = {
            f: {
                "all_points": np.arange(
                    df_sense.loc[f, "lower_bound_point"],
                    df_sense.loc[f, "upper_bound_point"] + 1e-3,
                    1,
                ),
            }
            for f in pool_top_features
        }

        best_metric = 1e9

        _count = 0
        # For all combinations of nb_max_feature from the set of selected binary features
        nb_combi = len(
            list(itertools.combinations(pool_top_features, self.nb_max_features))
        )

        # Max number of point combination
        dim_max_point_combination = (
            max(len(dict_point_ranges[f]["all_points"]) for f in pool_top_features)
            ** self.nb_max_features
        )

        # Compute cube of all points combination :
        # number of k feature among n x cardinality of feature pool x max number of point combination for k features
        cube = np.zeros(
            shape=(nb_combi, len(pool_top_features), dim_max_point_combination),
            dtype=np.float16,
        )

        end_time_P1 = time.time()
        print("Preparation:", end_time_P1 - start_time_P1)

        start_time_P2 = time.time()

        # for each feature combination, compute point combinations
        for i, top_features in enumerate(
            itertools.combinations(pool_top_features, self.nb_max_features)
        ):
            # Gather all point ranges for each binary feature
            all_points_by_feature = [
                dict_point_ranges[f]["all_points"] if f in top_features else [0]
                for f in pool_top_features
            ]

            # Compute the cartesian product of all possible point values for each feature
            # This creates a nxd matrix with n being the number of combinations of points for each binary feature
            # and d being the number of selected binary features
            all_points_possibilities = np.array(
                list(itertools.product(*all_points_by_feature))
            )
            all_points_possibilities = all_points_possibilities.T
            #     all_points_possibilities = all_points_possibilities.reshape(len(pool_top_features),-1)
            cube[i, :, : all_points_possibilities.shape[-1]] = all_points_possibilities

        end_time_P2 = time.time()
        print("Preparation compute:", end_time_P2 - start_time_P2)

        if chunk_size_cube is None:
            chunk_size_cube = (50, len(pool_top_features), 20)

        start_time_P3 = time.time()

        dask_cube = da.from_array(cube, chunks=chunk_size_cube)
        dataset_transpose = X_binarized[list(pool_top_features)].values.T

        if chunk_size_data is None:
            chunk_size_data = dataset_transpose.shape
        dask_dataset_T = da.from_array(dataset_transpose, chunks=chunk_size_data)

        dask_cube_augmented = dask.array.einsum(
            "ijk,jl->ijkl",
            dask_cube,
            dask_dataset_T,
            optimize="optimal",
            dtype=np.int8,
            casting="unsafe",
        )

        dask_score_all_case = dask_cube_augmented.sum(axis=1, dtype=np.int8).compute()
        end_time_P3 = time.time()
        print("Dask compute:", end_time_P3 - start_time_P3)

        start_time_P4 = time.time()

        auc_results = np.zeros(
            shape=(dask_score_all_case.shape[0], dask_score_all_case.shape[1])
        )
        for i in range(dask_score_all_case.shape[0]):
            for j in range(dask_score_all_case.shape[1]):
                auc_results[i, j] = fast_numba_auc(
                    y.values, y_score=dask_score_all_case[i, j, :]
                )
                # fast_numba_auc(
                #     y.values, y_score=dask_score_all_case[i, j, :]
                # )

        flatten_max_index = auc_results.argmax()
        idx_max = np.unravel_index(flatten_max_index, auc_results.shape)
        best_metric = auc_results[idx_max[0], idx_max[1]]
        best_points = cube[idx_max[0], :, idx_max[1]]
        best_feature_and_point_selection = [
            (point, f)
            for point, f in zip(best_points, list(pool_top_features))
            if point != 0
        ]
        best_scenario, best_feature = zip(*best_feature_and_point_selection)
        best_metric, best_scenario, best_feature

        best_scenario, best_feature_selection = zip(*best_feature_and_point_selection)
        best_feature_selection = list(best_feature_selection)

        end_time_P4 = time.time()
        print("AUC:", end_time_P4 - start_time_P4)

        start_time_P5 = time.time()

        df_calibration = pd.DataFrame(columns=["target", "score"])
        df_calibration["target"] = y_calib.values
        df_calibration["score"] = np.matmul(
            X_calib_binarized[best_feature_selection].values, best_scenario
        )

        df_score_proba_association = df_calibration.groupby("score")["target"].mean()
        df_score_proba_association.columns = ["proba"]
        min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
        max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))
        full_index = np.arange(min_sum_point, max_sum_point + 1e-3)
        missing_index = set(full_index) - set(df_score_proba_association.index)

        df_score_proba_association = df_score_proba_association.reindex(full_index)
        df_score_proba_association = df_score_proba_association.interpolate(
            method="linear"
        )

        df_reordering = pd.DataFrame(df_score_proba_association.copy())

        # Count cardinality of each score
        df_reordering["count"] = df_calibration.groupby("score")["target"].count()
        df_reordering["count"] = df_reordering["count"].fillna(0)
        df_reordering["target"] = df_reordering["target"].fillna(0)

        # Adjust probability to have an optimized logloss and calibration
        df_cvx = df_reordering.copy()

        # Compute number of positive and negative samples at each score value
        positive_sample_count = (
            df_cvx["target"].values * df_cvx["count"].values
        ).astype(int)
        negative_sample_count = df_cvx["count"].values - positive_sample_count

        # Declare the list of probabilities to be set as variables
        list_proba = [cp.Variable(1) for _ in df_cvx.index]

        # Compute total size of samples to normalize the logloss
        total_count = df_cvx["count"].sum()

        # Compute the logloss at each score in a list in order to sum it later
        # the logloss at each score is simple as all samples will have the same
        # probability p. for all positive samples, add -log(p), for all negative samples add -log(1-p)
        list_expression = [
            -cp.log(p) * w_pos - cp.log(1 - p) * w_neg
            for p, w_pos, w_neg in zip(
                list_proba, positive_sample_count, negative_sample_count
            )
        ]
        objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objective

        # Declare the constraints for the probabilities
        # the probability at each score should be higher than probabilities at a lower score
        constraints = []
        for p in list_proba:
            constraints.append(p >= 0)
            constraints.append(p <= 1)
        for i in range(1, len(list_proba)):
            # TODO : Put the threshold away and combine all similar scores into 1
            constraints.append(list_proba[i] - list_proba[i - 1] - 1e-3 >= 0)

        problem = cp.Problem(objective, constraints)

        opt = problem.solve(verbose=False)

        # Get the optimized value for the probabilities
        df_reordering["sorted_proba"] = [p.value[0] for p in list_proba]

        self.df_reordering_debug = df_reordering

        end_time_P5 = time.time()
        print("calibration:", end_time_P5 - start_time_P5)

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=best_feature_selection)
        self.feature_point_card[self._POINT_COL] = best_scenario

        self.feature_point_card["binary_feature"] = self.feature_point_card.index.values
        self.feature_point_card[
            self._FEATURE_COL
        ] = self.feature_point_card.index.values

        self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
        self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
            "binary_feature"
        ].values

        possible_scores = list(df_reordering.index)
        possible_risks = df_reordering["sorted_proba"]
        # Compute nice display of probabilities
        possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

        # Assign dataframe to score_card attribute
        self.score_card = pd.DataFrame(
            index=["SCORE", "RISK", "_RISK_FLOAT"],
            data=[possible_scores, possible_risks_pct, possible_risks],
        )
        self._total_fit_time = time.time() - start_time
        self._model_fit_time = self._total_fit_time - self.binarizer._fit_time


class EBMRiskScore(RiskScore):
    """
    Risk score model based on an EBM binarizer, ranking of features and exhaustive enumeration.

    This class is a child class of _BaseRiskScore. It implements the fit method that creates the feature-point card and score card attribute.
    It computes them by binarizing features based on EBM, ranking binary features and doing an exhaustive enumeration of features combination.
    It performs a hierarchical optimization:
    1) first it optimizes ROC AUC and/or PR AUC by selecting points for all combinations of binary feature;
    2) once the binary feature combination is chosen with the corresponding interger points, the logloss is optimized for each possible sum of points.

    Attributes
    ----------
    max_number_binaries_by_features : int
        maximum number of binary features to compute by feature
    nb_max_features: int
        maximum number of binary features to select
    min_point_value : int
        minimum points to assign for a binary feature
    max_point_value: ExplainableBoostingClassifier
        maximum points to assign for a binary feature
    _df_info: pandas.DataFrame
        Dataframe containing the link between a binary feature and its origin



    Methods
    -------
    fit(self, X, y):
        function creating the feature-point card and score card attributes via Optuna



    From _BaseRiskScore:

    @staticmethod
    _predict_proba_score(score, intercept):
        computes the value of the logistic function at score+intercept value

    predict(self, X, proba_threshold=0.5):
        function that predicts the positive/negative class by using a threshold on the probability given by the model

    predict_proba(self, X):
        function that predicts the probability of the positive class

    summary(self):
        function that prints the feature-point card and score card of the model
    """

    def __init__(
        self,
        nb_max_features: int = 4,
        min_point_value: int = -2,
        max_point_value: int = 3,
        max_number_binaries_by_features: int = 3,
        nb_additional_features: Optional[int] = 4,
        df_info: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            nb_max_features (int): Number of maximum binary features to be selected
            min_point_value (int, optional): Minimum point assigned to a binary feature. Defaults to -2.
            max_point_value (int, optional): Maximum point assigned to a binary feature. Defaults to 3.
            max_number_binaries_by_features (int, optional): Maximum number of binary features by original feature. Defaults to 3.
            nb_additional_features (int, optional): Number of additional features to consider for the binary feature selection. Defaults to 3.
            df_info (pandas.DataFrame, optional): DataFrame linking original feature and binary feature. Defaults to None.
        """
        super().__init__(
            binarizer=EBMBinarizer(
                max_number_binaries_by_features=max_number_binaries_by_features
            ),
            nb_max_features=nb_max_features,
            min_point_value=min_point_value,
            max_point_value=max_point_value,
            nb_additional_features=nb_additional_features,
            df_info=df_info,
        )

        # # TODO Check value
        # self.max_number_binaries_by_features: int = max_number_binaries_by_features
        # self._binarizer = AutoBinarizer(
        #     max_number_binaries_by_features=self.max_number_binaries_by_features
        # )
        # self.nb_additional_features = int(nb_additional_features)

    # def fit(
    #     self,
    #     X: pd.DataFrame,
    #     y: pd.Series,
    #     ranker,
    #     calibrator,
    #     X_calib: pd.DataFrame = None,
    #     y_calib: pd.Series = None,
    #     categorical_features=None,
    #     fit_binarizer=True,
    #     enumeration_maximization_metric=fast_numba_auc,
    # ):
    #     """Function that search best parameters (choice of binary features, points and probabilities) of a risk score model.

    #     It computes them by binarizing features based on EBM, ranking binary features and doing an exhaustive enumeration of features combination.
    #     It performs a hierarchical optimization:
    #     1) first it optimizes ROC AUC and/or PR AUC by selecting points for all combinations of selected binary feature.
    #     The selection of binary features combination is done by:
    #         a) ranking the binary features,
    #         b) taking the top features according to the ranking,
    #         c) enumerate all combinations of binary features,
    #         d) enumerate all point assignment for each combination of binary features generated.
    #         e) choose the best combination of binary feature and point based on a ranking metric
    #     Different ranking techniques are available for step b) :
    #     LogOddsDensity, DiverseLogOddsDensity, CumulativeMetric, BordaRank, LassoPathRank, LarsPathRank, OMPRank, FasterRiskRank

    #     2) once the binary feature combination is chosen with the corresponding interger points, the logloss is optimized for each possible sum of points.
    #     The logloss optimization can be done on a different dataset (X_calib, y_calib).
    #     It can be done with a vanilla mode (preferred when train or calibration set is big enough), or a bootstrapped mode to avoid overfitting when there are few samples.
    #     These logloss optimizer with more details can be found in the calibration script of the package.

    #     Args:
    #         X (pandas.DataFrame): Dataset of features to fit the risk score model on
    #         y (pandas.Series): Target binary values
    #         X_calib (pandas.DataFrame): Dataset of features to calibrate probabilities on
    #         y_calib (pandas.Series): Target binary values for calibration
    #         categorical_features: list of categorical features for the binarizer
    #         fit_binarizer: boolean to indicate the binarizer should be fitted or not
    #         sorting_method: ranking method of features #TODO change to ranker object
    #         optimization_method: optimization method to define probabilities (logloss)
    #     """
    #     start_time = time.time()

    #     # Binarize the features with the AutoBinarizer class
    #     if fit_binarizer:
    #         self._binarizer.fit(X, y, categorical_features=categorical_features)
    #     df_info = self._binarizer.df_score_feature

    #     start_time_P1 = time.time()

    #     # Prepare calibration set
    #     if X_calib is None:
    #         X_calib = X.copy()
    #         y_calib = y.copy()

    #     # Binarize the features with the previously fitted binarizer
    #     X_binarized = self._binarizer.transform(X)
    #     X_calib_binarized = self._binarizer.transform(X_calib)

    #     fit_transform_time = time.time() - start_time

    #     # Rank the binary feature by likeliness to be important for the risk score model

    #     self._binarizer.old_df_info = df_info.copy()

    #     df_ranker = df_info[["log_odds", "density", "feature"]].copy()
    #     # df_ranker.columns = ["log_odds", "density", "feature"]
    #     df_rank = ranker.compute_ranking_features(
    #         df=df_ranker,
    #         X_binarized=X_binarized,
    #         y=y,
    #         nb_steps=(self.nb_additional_features + self.nb_max_features),
    #     )

    #     df_info = df_info.merge(
    #         df_rank, right_index=True, left_index=True, how="left", validate="1:1"
    #     )
    #     df_info = df_info.sort_values(by="rank", ascending=True)

    #     pool_top_features = df_info.index[
    #         : self.nb_max_features + self.nb_additional_features
    #     ]

    #     # Compute bounds for points for each feature to reduce optimization space
    #     # -> negative points for negative log odds
    #     # -> positive points for positive log odds
    #     df_sense = df_info.iloc[
    #         : self.nb_max_features + self.nb_additional_features
    #     ].copy()

    #     df_sense.loc[:, "lower_bound_point"] = np.where(
    #         df_sense["log_odds"] > 0,
    #         min([1, self.max_point_value]),
    #         self.min_point_value,
    #     )

    #     df_sense.loc[:, "upper_bound_point"] = np.where(
    #         df_sense["log_odds"] > 0,
    #         self.max_point_value,
    #         max([-1, self.min_point_value]),
    #     )

    #     # Define all possible integer values for each binary feature
    #     dict_point_ranges = {
    #         f: {
    #             "all_points": np.arange(
    #                 df_sense.loc[f, "lower_bound_point"],
    #                 df_sense.loc[f, "upper_bound_point"] + 1e-3,
    #                 1,
    #             ),
    #         }
    #         for f in pool_top_features
    #     }

    #     best_metric = 1e9

    #     _count = 0
    #     # For all combinations of nb_max_feature from the set of selected binary features
    #     nb_combi = len(
    #         list(itertools.combinations(pool_top_features, self.nb_max_features))
    #     )

    #     # Max number of point combination
    #     dim_max_point_combination = (
    #         max(len(dict_point_ranges[f]["all_points"]) for f in pool_top_features)
    #         ** self.nb_max_features
    #     )

    #     # Compute cube of all points combination :
    #     # number of combinations of self.nb_max_features feature among the selected pool of binary features x cardinality of feature pool
    #     cube = np.zeros(
    #         shape=(nb_combi * dim_max_point_combination, len(pool_top_features)),
    #         dtype=np.float16,
    #     )

    #     end_time_P1 = time.time()

    #     start_time_P2 = time.time()

    #     idx = 0
    #     # for each feature combination, compute point combinations
    #     for i, top_features in enumerate(
    #         itertools.combinations(pool_top_features, self.nb_max_features)
    #     ):
    #         # Gather all point ranges for each binary feature
    #         # if the feature is not in the current selected combination, the only point possibility is 0
    #         all_points_by_feature = [
    #             dict_point_ranges[f]["all_points"] if f in top_features else [0]
    #             for f in pool_top_features
    #         ]

    #         # Compute the cartesian product of all possible point values for each feature
    #         # This creates a nxd matrix with n being the number of combinations of points for each binary feature
    #         # and d being the total number of binary features in pool_top_features.
    #         # for each line, there are only self.nb_max_features non zero point values.
    #         all_points_possibilities = np.array(
    #             list(itertools.product(*all_points_by_feature))
    #         )

    #         # Put this points combination into the global enumeration of points
    #         cube[
    #             idx : idx + all_points_possibilities.shape[0], :
    #         ] = all_points_possibilities
    #         idx += all_points_possibilities.shape[0]

    #     end_time_P2 = time.time()

    #     # if chunk_size_cube is None:
    #     chunk_size_cube = (50, len(pool_top_features), 20)

    #     start_time_P4 = time.time()

    #     # Use dask to compute the metric to optimize on all points combination

    #     # Dask array for the global enumeration of combinations
    #     dask_cube = da.from_array(cube, chunks=((50, len(pool_top_features))))

    #     # Dask array for the dataset of selected binary features
    #     dataset_transpose = X_binarized[list(pool_top_features)].values.T
    #     dask_dataset_T = da.from_array(
    #         dataset_transpose, chunks=dataset_transpose.shape
    #     )

    #     # 1Dfication of maximization metric function

    #     enumeration_optimization_metric_1d = lambda y_score: [
    #         enumeration_maximization_metric(y.values, y_score)
    #     ]

    #     # computation of score for all samples for all enumerated points combination
    #     matmul = da.matmul(dask_cube, dask_dataset_T)

    #     # computation of the maximum value on the defined enumeration metric
    #     idx_max = da.argmax(
    #         da.apply_along_axis(
    #             enumeration_optimization_metric_1d,
    #             axis=1,
    #             arr=matmul,
    #             shape=(1,),
    #             dtype=matmul.dtype,
    #         )
    #     ).compute()

    #     # Getting the points combination with the maximumu value
    #     best_points = cube[idx_max, :]
    #     best_feature_and_point_selection = [
    #         (point, f)
    #         for point, f in zip(best_points, list(pool_top_features))
    #         if point != 0
    #     ]

    #     best_scenario, best_feature_selection = zip(*best_feature_and_point_selection)
    #     best_feature_selection = list(best_feature_selection)

    #     end_time_P4 = time.time()

    #     # Probabilities computation
    #     start_time_P5 = time.time()

    #     df_calibration = pd.DataFrame(columns=["target", "score"])
    #     df_calibration["target"] = y_calib.values
    #     df_calibration["score"] = np.matmul(
    #         X_calib_binarized[best_feature_selection].values, best_scenario
    #     )

    #     min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
    #     max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))

    #     calibrator.calibrate(df_calibration, min_sum_point, max_sum_point)

    #     df_reordering = calibrator.calibrate(
    #         df_calibration, min_sum_point, max_sum_point
    #     )

    #     end_time_P5 = time.time()

    #     # Build feature-point card
    #     self.feature_point_card = pd.DataFrame(index=best_feature_selection)
    #     self.feature_point_card[self._POINT_COL] = best_scenario

    #     self.feature_point_card["binary_feature"] = self.feature_point_card.index.values
    #     self.feature_point_card[
    #         self._FEATURE_COL
    #     ] = self.feature_point_card.index.values

    #     self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
    #     self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
    #         "binary_feature"
    #     ].values

    #     possible_scores = list(df_reordering.index)
    #     possible_risks = df_reordering["sorted_proba"]
    #     # Compute nice display of probabilities
    #     possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

    #     # Assign dataframe to score_card attribute
    #     self.score_card = pd.DataFrame(
    #         index=["SCORE", "RISK", "_RISK_FLOAT"],
    #         data=[possible_scores, possible_risks_pct, possible_risks],
    #     )
    #     self._total_fit_time = time.time() - start_time
    #     self._model_fit_time = self._total_fit_time - fit_transform_time

    # def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
    #     X_binarized = self._binarizer.transform(X)
    #     return super().predict_proba(X_binarized)

    # def _fit_numpy(
    #     self,
    #     X: pd.DataFrame,
    #     y: pd.Series,
    #     X_calib: pd.DataFrame = None,
    #     y_calib: pd.Series = None,
    #     categorical_features=None,
    # ):
    #     start_time = time.time()
    #     # Binarize the features with the AutoBinarizer class
    #     self._binarizer.fit(X, y, categorical_features=categorical_features)
    #     df_info = self._binarizer.df_score_feature

    #     if X_calib is None:
    #         X_calib = X.copy()
    #         y_calib = y.copy()
    #     X_binarized = self._binarizer.transform(X)
    #     X_calib_binarized = self._binarizer.transform(X_calib)

    #     # Rank the binary feature by likeliness to be important for the risk score model
    #     # The current estimated importance is the log odd computed by the EBM model x number of positive samples for
    #     # that binary feature.
    #     # The cardinality of the positive samples is de emphasize by taking the 0.95 power.
    #     # TODO: take into account the impact of having mixed class in the samples

    #     df_info["abs_contribution"] = df_info["log_odds"].abs() * df_info[
    #         "density"
    #     ].fillna(0).astype(int).pow(
    #         0.95
    #     )  # to de emphasize large values impact

    #     # Compute the reduced pool of top features to choose from
    #     pool_top_features = df_info.sort_values(
    #         by="abs_contribution", ascending=False
    #     ).index[: self.nb_max_features + self.nb_additional_features]

    #     # Compute bounds for points for each feature to reduce optimization space
    #     # -> negative points for negative log odds
    #     # -> positive points for positive log odds
    #     df_sense = df_info.sort_values(by="abs_contribution", ascending=False).iloc[
    #         : self.nb_max_features + self.nb_additional_features
    #     ]
    #     df_sense["lower_bound_point"] = np.where(
    #         df_sense["log_odds"] > 0,
    #         min([1, self.max_point_value]),
    #         self.min_point_value,
    #     )
    #     df_sense["upper_bound_point"] = np.where(
    #         df_sense["log_odds"] > 0,
    #         self.max_point_value,
    #         max([-1, self.min_point_value]),
    #     )

    #     # Define all possible integer values for each binary feature
    #     dict_point_ranges = {
    #         f: {
    #             "all_points": np.arange(
    #                 df_sense.loc[f, "lower_bound_point"],
    #                 df_sense.loc[f, "upper_bound_point"] + 1e-3,
    #                 1,
    #             ),
    #         }
    #         for f in pool_top_features
    #     }

    #     best_metric = 1e9

    #     _count = 0
    #     # For all combinations of nb_max_feature from the set of selected binary features
    #     nb_combi = len(
    #         list(itertools.combinations(pool_top_features, self.nb_max_features))
    #     )

    #     # Max number of point combination
    #     dim_max_point_combination = (
    #         max(len(dict_point_ranges[f]["all_points"]) for f in pool_top_features)
    #         ** self.nb_max_features
    #     )

    #     # Compute cube of all points combination :
    #     # number of k feature among n x cardinality of feature pool x max number of point combination for k features
    #     cube = np.zeros(
    #         shape=(nb_combi, len(pool_top_features), dim_max_point_combination),
    #         dtype=np.float16,
    #     )

    #     # for each feature combination, compute point combinations
    #     for i, top_features in enumerate(
    #         itertools.combinations(pool_top_features, self.nb_max_features)
    #     ):
    #         # Gather all point ranges for each binary feature
    #         all_points_by_feature = [
    #             dict_point_ranges[f]["all_points"] if f in top_features else [0]
    #             for f in pool_top_features
    #         ]

    #         # Compute the cartesian product of all possible point values for each feature
    #         # This creates a nxd matrix with n being the number of combinations of points for each binary feature
    #         # and d being the number of selected binary features
    #         all_points_possibilities = np.array(
    #             list(itertools.product(*all_points_by_feature))
    #         )
    #         all_points_possibilities = all_points_possibilities.T
    #         #     all_points_possibilities = all_points_possibilities.reshape(len(pool_top_features),-1)
    #         cube[i, :, : all_points_possibilities.shape[-1]] = all_points_possibilities

    #     # TODO uncomment for the non dask version
    #     cube_augmented = np.einsum(
    #         "ijk,jl->ijkl",
    #         cube,
    #         X_binarized[list(pool_top_features)].values.T,
    #         optimize="optimal",
    #         dtype=np.int8,
    #         casting="unsafe",
    #     )

    #     # score for each feature combi, for each point possibilities, for each sample
    #     score_all_case = cube_augmented.sum(axis=1)

    #     auc_results = np.zeros(shape=(score_all_case.shape[0], score_all_case.shape[1]))
    #     for i in range(score_all_case.shape[0]):
    #         for j in range(score_all_case.shape[1]):
    #             auc_results[i, j] = fast_numba_auc(
    #                 y.values, y_score=score_all_case[i, j, :]
    #             )

    #     flatten_max_index = auc_results.argmax()
    #     idx_max = np.unravel_index(flatten_max_index, auc_results.shape)
    #     best_metric = auc_results[idx_max[0], idx_max[1]]
    #     best_points = cube[idx_max[0], :, idx_max[1]]
    #     best_feature_and_point_selection = [
    #         (point, f)
    #         for point, f in zip(best_points, list(pool_top_features))
    #         if point != 0
    #     ]
    #     best_scenario, best_feature_selection = zip(*best_feature_and_point_selection)
    #     best_feature_selection = list(best_feature_selection)

    #     df_calibration = pd.DataFrame(columns=["target", "score"])
    #     df_calibration["target"] = y_calib.values
    #     df_calibration["score"] = np.matmul(
    #         X_calib_binarized[best_feature_selection].values, best_scenario
    #     )

    #     df_score_proba_association = df_calibration.groupby("score")["target"].mean()
    #     df_score_proba_association.columns = ["proba"]
    #     min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
    #     max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))
    #     full_index = np.arange(min_sum_point, max_sum_point + 1e-3)
    #     missing_index = set(full_index) - set(df_score_proba_association.index)

    #     df_score_proba_association = df_score_proba_association.reindex(full_index)
    #     df_score_proba_association = df_score_proba_association.interpolate(
    #         method="linear"
    #     )

    #     df_reordering = pd.DataFrame(df_score_proba_association.copy())
    #     # Count cardinality of each score
    #     df_reordering["count"] = df_calibration.groupby("score")["target"].count()
    #     df_reordering["count"] = df_reordering["count"].fillna(0)
    #     df_reordering["target"] = df_reordering["target"].fillna(0)

    #     # Adjust probability to have an optimized logloss and calibration
    #     df_cvx = df_reordering.copy()

    #     # Compute number of positive and negative samples at each score value
    #     positive_sample_count = (
    #         df_cvx["target"].values * df_cvx["count"].values
    #     ).astype(int)
    #     negative_sample_count = df_cvx["count"].values - positive_sample_count

    #     # Declare the list of probabilities to be set as variables
    #     list_proba = [cp.Variable(1) for _ in df_cvx.index]

    #     # Compute total size of samples to normalize the logloss
    #     total_count = df_cvx["count"].sum()

    #     # Compute the logloss at each score in a list in order to sum it later
    #     # the logloss at each score is simple as all samples will have the same
    #     # probability p. for all positive samples, add -log(p), for all negative samples add -log(1-p)
    #     list_expression = [
    #         -cp.log(p) * w_pos - cp.log(1 - p) * w_neg
    #         for p, w_pos, w_neg in zip(
    #             list_proba, positive_sample_count, negative_sample_count
    #         )
    #     ]
    #     objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objective

    #     # Declare the constraints for the probabilities
    #     # the probability at each score should be higher than probabilities at a lower score
    #     constraints = []
    #     for p in list_proba:
    #         constraints.append(p >= 0)
    #         constraints.append(p <= 1)
    #     for i in range(1, len(list_proba)):
    #         # TODO : Put the threshold away and combine all similar scores into 1
    #         constraints.append(list_proba[i] - list_proba[i - 1] - 1e-3 >= 0)

    #     problem = cp.Problem(objective, constraints)

    #     opt = problem.solve(verbose=False)

    #     # Get the optimized value for the probabilities
    #     df_reordering["sorted_proba"] = [p.value[0] for p in list_proba]

    #     self.df_reordering_debug = df_reordering

    #     # Build feature-point card
    #     self.feature_point_card = pd.DataFrame(index=best_feature_selection)
    #     self.feature_point_card[self._POINT_COL] = best_scenario

    #     self.feature_point_card["binary_feature"] = self.feature_point_card.index.values
    #     self.feature_point_card[
    #         self._FEATURE_COL
    #     ] = self.feature_point_card.index.values

    #     self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
    #     self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
    #         "binary_feature"
    #     ].values

    #     possible_scores = list(df_reordering.index)
    #     possible_risks = df_reordering["sorted_proba"]
    #     # Compute nice display of probabilities
    #     possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

    #     # Assign dataframe to score_card attribute
    #     self.score_card = pd.DataFrame(
    #         index=["SCORE", "RISK", "_RISK_FLOAT"],
    #         data=[possible_scores, possible_risks_pct, possible_risks],
    #     )
    #     self._total_fit_time = time.time() - start_time
    #     self._model_fit_time = self._total_fit_time - self._binarizer._fit_time

    # def _fit_dask_old(
    #     self,
    #     X: pd.DataFrame,
    #     y: pd.Series,
    #     X_calib: pd.DataFrame = None,
    #     y_calib: pd.Series = None,
    #     categorical_features=None,
    #     chunk_size_cube=None,
    #     chunk_size_data=None,
    #     fit_binarizer=True,
    # ):
    #     start_time = time.time()

    #     # Binarize the features with the AutoBinarizer class
    #     if fit_binarizer:
    #         self._binarizer.fit(X, y, categorical_features=categorical_features)
    #     df_info = self._binarizer.df_score_feature

    #     start_time_P1 = time.time()
    #     if X_calib is None:
    #         X_calib = X.copy()
    #         y_calib = y.copy()
    #     X_binarized = self._binarizer.transform(X)
    #     X_calib_binarized = self._binarizer.transform(X_calib)

    #     fit_transform_time = time.time() - start_time
    #     # Rank the binary feature by likeliness to be important for the risk score model
    #     # The current estimated importance is the log odd computed by the EBM model x number of positive samples for
    #     # that binary feature.
    #     # The cardinality of the positive samples is de emphasize by taking the 0.95 power.
    #     # TODO: take into account the impact of having mixed class in the samples

    #     df_info["abs_contribution"] = df_info["log_odds"].abs() * df_info[
    #         "density"
    #     ].fillna(0).astype(int).pow(
    #         0.95
    #     )  # to de emphasize large values impact

    #     # Compute the reduced pool of top features to choose from
    #     pool_top_features = df_info.sort_values(
    #         by="abs_contribution", ascending=False
    #     ).index[: self.nb_max_features + self.nb_additional_features]

    #     # Compute bounds for points for each feature to reduce optimization space
    #     # -> negative points for negative log odds
    #     # -> positive points for positive log odds
    #     df_sense = df_info.sort_values(by="abs_contribution", ascending=False).iloc[
    #         : self.nb_max_features + self.nb_additional_features
    #     ]
    #     df_sense["lower_bound_point"] = np.where(
    #         df_sense["log_odds"] > 0,
    #         min([1, self.max_point_value]),
    #         self.min_point_value,
    #     )
    #     df_sense["upper_bound_point"] = np.where(
    #         df_sense["log_odds"] > 0,
    #         self.max_point_value,
    #         max([-1, self.min_point_value]),
    #     )

    #     # Define all possible integer values for each binary feature
    #     dict_point_ranges = {
    #         f: {
    #             "all_points": np.arange(
    #                 df_sense.loc[f, "lower_bound_point"],
    #                 df_sense.loc[f, "upper_bound_point"] + 1e-3,
    #                 1,
    #             ),
    #         }
    #         for f in pool_top_features
    #     }

    #     best_metric = 1e9

    #     _count = 0
    #     # For all combinations of nb_max_feature from the set of selected binary features
    #     nb_combi = len(
    #         list(itertools.combinations(pool_top_features, self.nb_max_features))
    #     )

    #     # Max number of point combination
    #     dim_max_point_combination = (
    #         max(len(dict_point_ranges[f]["all_points"]) for f in pool_top_features)
    #         ** self.nb_max_features
    #     )

    #     # Compute cube of all points combination :
    #     # number of k feature among n x cardinality of feature pool x max number of point combination for k features
    #     cube = np.zeros(
    #         shape=(nb_combi, len(pool_top_features), dim_max_point_combination),
    #         dtype=np.float16,
    #     )

    #     end_time_P1 = time.time()
    #     print("Preparation:", end_time_P1 - start_time_P1)

    #     start_time_P2 = time.time()

    #     # for each feature combination, compute point combinations
    #     for i, top_features in enumerate(
    #         itertools.combinations(pool_top_features, self.nb_max_features)
    #     ):
    #         # Gather all point ranges for each binary feature
    #         all_points_by_feature = [
    #             dict_point_ranges[f]["all_points"] if f in top_features else [0]
    #             for f in pool_top_features
    #         ]

    #         # Compute the cartesian product of all possible point values for each feature
    #         # This creates a nxd matrix with n being the number of combinations of points for each binary feature
    #         # and d being the number of selected binary features
    #         all_points_possibilities = np.array(
    #             list(itertools.product(*all_points_by_feature))
    #         )
    #         all_points_possibilities = all_points_possibilities.T
    #         #     all_points_possibilities = all_points_possibilities.reshape(len(pool_top_features),-1)
    #         cube[i, :, : all_points_possibilities.shape[-1]] = all_points_possibilities

    #     end_time_P2 = time.time()
    #     print("Preparation compute:", end_time_P2 - start_time_P2)

    #     if chunk_size_cube is None:
    #         chunk_size_cube = (50, len(pool_top_features), 20)

    #     start_time_P3 = time.time()

    #     dask_cube = da.from_array(cube, chunks=chunk_size_cube)
    #     dataset_transpose = X_binarized[list(pool_top_features)].values.T

    #     if chunk_size_data is None:
    #         chunk_size_data = dataset_transpose.shape
    #     dask_dataset_T = da.from_array(dataset_transpose, chunks=chunk_size_data)

    #     dask_cube_augmented = dask.array.einsum(
    #         "ijk,jl->ijkl",
    #         dask_cube,
    #         dask_dataset_T,
    #         optimize="optimal",
    #         dtype=np.int8,
    #         casting="unsafe",
    #     )

    #     dask_score_all_case = dask_cube_augmented.sum(axis=1, dtype=np.int8).compute()
    #     end_time_P3 = time.time()
    #     print("Dask compute:", end_time_P3 - start_time_P3)

    #     start_time_P4 = time.time()

    #     auc_results = np.zeros(
    #         shape=(dask_score_all_case.shape[0], dask_score_all_case.shape[1])
    #     )
    #     for i in range(dask_score_all_case.shape[0]):
    #         for j in range(dask_score_all_case.shape[1]):
    #             auc_results[i, j] = fast_numba_auc(
    #                 y.values, y_score=dask_score_all_case[i, j, :]
    #             )
    #             # fast_numba_auc(
    #             #     y.values, y_score=dask_score_all_case[i, j, :]
    #             # )

    #     flatten_max_index = auc_results.argmax()
    #     idx_max = np.unravel_index(flatten_max_index, auc_results.shape)
    #     best_metric = auc_results[idx_max[0], idx_max[1]]
    #     best_points = cube[idx_max[0], :, idx_max[1]]
    #     best_feature_and_point_selection = [
    #         (point, f)
    #         for point, f in zip(best_points, list(pool_top_features))
    #         if point != 0
    #     ]
    #     best_scenario, best_feature = zip(*best_feature_and_point_selection)
    #     best_metric, best_scenario, best_feature

    #     best_scenario, best_feature_selection = zip(*best_feature_and_point_selection)
    #     best_feature_selection = list(best_feature_selection)

    #     end_time_P4 = time.time()
    #     print("AUC:", end_time_P4 - start_time_P4)

    #     start_time_P5 = time.time()

    #     df_calibration = pd.DataFrame(columns=["target", "score"])
    #     df_calibration["target"] = y_calib.values
    #     df_calibration["score"] = np.matmul(
    #         X_calib_binarized[best_feature_selection].values, best_scenario
    #     )

    #     df_score_proba_association = df_calibration.groupby("score")["target"].mean()
    #     df_score_proba_association.columns = ["proba"]
    #     min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
    #     max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))
    #     full_index = np.arange(min_sum_point, max_sum_point + 1e-3)
    #     missing_index = set(full_index) - set(df_score_proba_association.index)

    #     df_score_proba_association = df_score_proba_association.reindex(full_index)
    #     df_score_proba_association = df_score_proba_association.interpolate(
    #         method="linear"
    #     )

    #     df_reordering = pd.DataFrame(df_score_proba_association.copy())

    #     # Count cardinality of each score
    #     df_reordering["count"] = df_calibration.groupby("score")["target"].count()
    #     df_reordering["count"] = df_reordering["count"].fillna(0)
    #     df_reordering["target"] = df_reordering["target"].fillna(0)

    #     # Adjust probability to have an optimized logloss and calibration
    #     df_cvx = df_reordering.copy()

    #     # Compute number of positive and negative samples at each score value
    #     positive_sample_count = (
    #         df_cvx["target"].values * df_cvx["count"].values
    #     ).astype(int)
    #     negative_sample_count = df_cvx["count"].values - positive_sample_count

    #     # Declare the list of probabilities to be set as variables
    #     list_proba = [cp.Variable(1) for _ in df_cvx.index]

    #     # Compute total size of samples to normalize the logloss
    #     total_count = df_cvx["count"].sum()

    #     # Compute the logloss at each score in a list in order to sum it later
    #     # the logloss at each score is simple as all samples will have the same
    #     # probability p. for all positive samples, add -log(p), for all negative samples add -log(1-p)
    #     list_expression = [
    #         -cp.log(p) * w_pos - cp.log(1 - p) * w_neg
    #         for p, w_pos, w_neg in zip(
    #             list_proba, positive_sample_count, negative_sample_count
    #         )
    #     ]
    #     objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objective

    #     # Declare the constraints for the probabilities
    #     # the probability at each score should be higher than probabilities at a lower score
    #     constraints = []
    #     for p in list_proba:
    #         constraints.append(p >= 0)
    #         constraints.append(p <= 1)
    #     for i in range(1, len(list_proba)):
    #         # TODO : Put the threshold away and combine all similar scores into 1
    #         constraints.append(list_proba[i] - list_proba[i - 1] - 1e-3 >= 0)

    #     problem = cp.Problem(objective, constraints)

    #     opt = problem.solve(verbose=False)

    #     # Get the optimized value for the probabilities
    #     df_reordering["sorted_proba"] = [p.value[0] for p in list_proba]

    #     self.df_reordering_debug = df_reordering

    #     end_time_P5 = time.time()
    #     print("calibration:", end_time_P5 - start_time_P5)

    #     # Build feature-point card
    #     self.feature_point_card = pd.DataFrame(index=best_feature_selection)
    #     self.feature_point_card[self._POINT_COL] = best_scenario

    #     self.feature_point_card["binary_feature"] = self.feature_point_card.index.values
    #     self.feature_point_card[
    #         self._FEATURE_COL
    #     ] = self.feature_point_card.index.values

    #     self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
    #     self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
    #         "binary_feature"
    #     ].values

    #     possible_scores = list(df_reordering.index)
    #     possible_risks = df_reordering["sorted_proba"]
    #     # Compute nice display of probabilities
    #     possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

    #     # Assign dataframe to score_card attribute
    #     self.score_card = pd.DataFrame(
    #         index=["SCORE", "RISK", "_RISK_FLOAT"],
    #         data=[possible_scores, possible_risks_pct, possible_risks],
    #     )
    #     self._total_fit_time = time.time() - start_time
    #     self._model_fit_time = self._total_fit_time - self._binarizer._fit_time
