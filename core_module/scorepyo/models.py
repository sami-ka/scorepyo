# mypy: ignore-errors
"""Classes to create and fit risk-score type model.
"""
import itertools
import numbers
import time
from abc import abstractmethod
from typing import Any, Optional

import dask.array as da
import numpy as np
import pandas as pd
import pandera as pa
from sklearn.exceptions import NotFittedError

from scorepyo._utils import fast_numba_auc
from scorepyo.binarizers import BinarizerProtocol, EBMBinarizer
from scorepyo.calibration import Calibrator, VanillaCalibrator
from scorepyo.exceptions import (
    MinPointOverMaxPointError,
    NegativeValueError,
    NonIntegerValueError,
)
from scorepyo.ranking import OMPRank, Ranker


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

    # @staticmethod
    # def _predict_proba_score(
    #     score: int, intercept: float, multiplier: Optional[float] = 1.0
    # ) -> np.ndarray:
    #     """Function that computes the logistic function value at score+intercept value

    #     Args:
    #         score (np.array(int)): sum of points coming from binary features
    #         intercept (float): intercept of score card. log-odds of having 0 point

    #     Returns:
    #         np.array(int): associated probability
    #     """
    #     score_associated_proba = 1 / (1 + np.exp(-(score + intercept) / multiplier))
    #     return score_associated_proba

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series, *args: Any, **kwargs: Any
    ) -> Optional[NotImplementedError]:
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
    Risk score model based on a binarizer, ranking of features, exhaustive enumeration with a maximization metric and a calibration method for probabilities.

    This class is a child class of _BaseRiskScore. It implements the fit method that creates the feature-point card and score card attribute.
    It computes them by binarizing features based on a given binarizer, ranking binary features and doing an exhaustive enumeration of features combination.
    It performs a hierarchical optimization:
    1) first it optimizes a metric based on scores. ROC AUC and/or PR AUC are good candidates as it is only based on the ranking of samples, compared to logloss which neeeds proper probabilities.
    The optimization is done by enumerating all possible selection of points for all combinations of binary feature and selecting the combination with the best value on the chosen metric;
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
    binarizer:
        binarizer object that transforms continuous and categorical features into binary features
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
        binarizer: BinarizerProtocol,
        nb_max_features: int = 4,
        min_point_value: int = -2,
        max_point_value: int = 3,
        max_number_binaries_by_features: int = 3,
        nb_additional_features: int = 4,
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
        self.binarizer: BinarizerProtocol = binarizer
        self.nb_additional_features: int = int(nb_additional_features)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ranker: Ranker = OMPRank(),
        calibrator: Calibrator = VanillaCalibrator(),
        X_calib: pd.DataFrame = None,
        y_calib: pd.Series = None,
        categorical_features="auto",
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
        Everyone can implement its own ranking technique, given that the customized ranker class implements the Ranker class

        2) once the binary feature combination is chosen with the corresponding interger points, the logloss is optimized for each possible sum of points.
        The logloss optimization can be done on a different dataset (X_calib, y_calib).
        It can be done with a vanilla mode (preferred when train or calibration set is big enough), or a bootstrapped mode to avoid overfitting when there are few samples.
        These logloss optimizer with more details can be found in the calibration script of the package.
        Everyone can implement its own calibration technique, given that the customized calibrator class implements the Calibrator class

        Args:
            X (pandas.DataFrame): Dataset of features to fit the risk score model on
            y (pandas.Series): Target binary values
            ranker (Ranker): Ranker object to rank binary features, see ranking.py
            calibrator (Calibrator): Calibrator object to define probabilities, see calibration.py
            X_calib (pandas.DataFrame): Dataset of features to calibrate probabilities on
            y_calib (pandas.Series): Target binary values for calibration
            categorical_features: list of categorical features for the binarizer
            fit_binarizer: boolean to indicate the binarizer should be fitted or not
            enumeration_maximization_metric: maximization function used for enumeration.
                This function needs to compute a maximization metric based on 2 arguments in this order :
                1) numpy array containing the binary target
                2) numpy array containing the predicted probability or score

        """
        start_time = time.time()

        # Binarize the features with the AutoBinarizer class
        if fit_binarizer:
            self.binarizer.fit(X, y, categorical_features=categorical_features)

        # TODO : check columns of df_info
        df_info = self.binarizer.get_info()

        # Prepare calibration set
        if X_calib is None:
            X_calib = X.copy()
            y_calib = y.copy()

        # Binarize the features with the previously fitted binarizer
        X_binarized = self.binarizer.transform(X)
        X_calib_binarized = self.binarizer.transform(X_calib)

        fit_transform_time = time.time() - start_time

        # Rank the binary feature by likeliness to be important for the risk score model

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

        idx = 0
        # for each feature combination, compute point combinations
        for _, top_features in enumerate(
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
                list(itertools.product(*all_points_by_feature))  # type: ignore
            )

            # Put this points combination into the global enumeration of points
            cube[
                idx : idx + all_points_possibilities.shape[0], :
            ] = all_points_possibilities
            idx += all_points_possibilities.shape[0]

        # Use dask to compute the metric to optimize on all points combination

        # Dask array for the global enumeration of combinations
        # TODO better design of chunk size
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

        # Probabilities computation

        df_calibration = pd.DataFrame(columns=["target", "score"])
        df_calibration["target"] = y_calib.values  # type: ignore
        df_calibration["score"] = np.matmul(
            X_calib_binarized[best_feature_selection].values, best_scenario
        )

        min_sum_point = np.sum(np.clip(best_scenario, a_max=0, a_min=None))
        max_sum_point = np.sum(np.clip(best_scenario, a_min=0, a_max=None))

        calibrator.calibrate(df_calibration, min_sum_point, max_sum_point)

        df_reordering = calibrator.calibrate(
            df_calibration, min_sum_point, max_sum_point
        )

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=best_feature_selection)
        self.feature_point_card[self._POINT_COL] = best_scenario
        self.feature_point_card.loc[:, self._POINT_COL] = self.feature_point_card[
            self._POINT_COL
        ].astype(int)
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

        self.score_card.loc["SCORE", :] = self.score_card.loc["SCORE", :].astype(int)
        self._total_fit_time = time.time() - start_time
        self._model_fit_time = self._total_fit_time - fit_transform_time

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_binarized = self.binarizer.transform(X)
        return super().predict_proba(X_binarized)


class EBMRiskScore(RiskScore):
    """
    Risk score model based on a EBMbinarizer, ranking of features, exhaustive enumeration with a maximization metric and a calibration method for probabilities.

    This class is a child class of _BaseRiskScore. It implements the fit method that creates the feature-point card and score card attribute.
    It computes them by binarizing features based on a given binarizer, ranking binary features and doing an exhaustive enumeration of features combination.
    It performs a hierarchical optimization:
    1) first it optimizes a metric based on scores. ROC AUC and/or PR AUC are good candidates as it is only based on the ranking of samples, compared to logloss which neeeds proper probabilities.
    The optimization is done by enumerating all possible selection of points for all combinations of binary feature and selecting the combination with the best value on the chosen metric;
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
    binarizer:
        binarizer object that transforms continuous and categorical features into binary features
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
