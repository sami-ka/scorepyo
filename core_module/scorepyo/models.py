"""Classes to create and fit risk-score type model.
"""
import itertools
import numbers
from abc import abstractmethod
from typing import Optional

import cvxpy as cp
import numpy as np
import optuna
import pandas as pd
import pandera as pa
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator

# from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from scorepyo.exceptions import (
    MinPointOverMaxPointError,
    NegativeValueError,
    NonIntegerValueError,
)
from scorepyo.preprocessing import AutoBinarizer


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


class OptunaRiskScore(_BaseRiskScore):
    """
    Risk score model based on Optuna.

    This class is a child class of _BaseRiskScore. It implements the fit method that creates the feature-point card and score card attribute.
    It computes them by leveraging the sampling efficiency of Optuna. Optuna is asked to select nb_max_features among all features, and assign points
    to each selected feature. It minimizes the logloss on a given dataset.


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
    fit(self, X, y):
        function creating the feature-point card and score card attributes via Optuna

    score_logloss_objective(self, trial, X, y):
        function that defines the logloss function used by Optuna


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
        df_info: Optional[pd.DataFrame] = None,
        optuna_optimize_params: Optional[dict] = None,
    ):
        """
        Args:
            nb_max_features (int): Number of maximum binary features to be selected
            min_point_value (int, optional): Minimum point assigned to a binary feature. Defaults to -2.
            max_point_value (int, optional): Maximum point assigned to a binary feature. Defaults to 3.
            df_info (pandas.DataFrame, optional): DataFrame linking original feature and binary feature. Defaults to None.
            optuna_optimize_params (dict, optional): parameters for optuna optimize function. Defaults to None.
        """
        super().__init__(nb_max_features, min_point_value, max_point_value, df_info)
        self.optuna_optimize_params = dict()
        if optuna_optimize_params is not None:
            self.optuna_optimize_params = optuna_optimize_params
        else:
            self.optuna_optimize_params["n_trials"] = 100
            self.optuna_optimize_params["timeout"] = 90

    def score_logloss_objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Logloss objective function for Risk score exploration parameters sampled with optuna.

        This function creates 2x`self.nb_max_features`+1 parameters for the optuna trial:
        - `self.nb_max_features` categorical parameters for the choice of binary features to build the risk score on
        - `self.nb_max_features` integer parameters for the choice of points associated to the selected binary feature
        - one float parameter for the intercept of the score card (i.e. the log odd associated to a score of 0)


        Args:
            trial (Optune.trial): Trial for optuna
            X (pandas.DataFrame): dataset of features to minimize scorecard logloss on
            y (nd.array): Target binary values

        Returns:
            float: log-loss value for risk score sampled parameters
        """
        # TODO : put option for no duplicate features
        # change way of doing the sampling by creating all combinations of binary features
        # https://stackoverflow.com/questions/73388133/is-there-a-way-for-optuna-suggest-categoricalto-return-multiple-choices-from-l

        # Define the parameters of trial for the selection of the subset of binary features respecting the maximum number
        param_grid_choice_feature = {
            f"feature_{i}_choice": trial.suggest_categorical(
                f"feature_{i}_choice", X.columns
            )
            for i in range(self.nb_max_features)
        }

        # # Define the parameters of trial for the points associated to each selected binary feature
        # param_grid_point = {
        #     f"feature_{i}_point": trial.suggest_int(
        #         f"feature_{i}_point", self.min_point_value, self.max_point_value
        #     )
        #     for i in range(self.nb_max_features)
        # }

        # Define the parameters of trial for the points associated to each selected binary feature
        non_zero_point = [
            i for i in range(self.min_point_value, self.max_point_value + 1) if i != 0
        ]
        param_grid_point = {
            f"feature_{i}_point": trial.suggest_categorical(
                f"feature_{i}_point", non_zero_point
            )
            for i in range(self.nb_max_features)
        }

        # Define the parameter of trial for the intercept of the risk score model
        score_intercept = trial.suggest_float(
            "intercept",
            -10 + self.min_point_value * self.nb_max_features,
            10 + self.max_point_value * self.nb_max_features,
            step=0.5,
        )

        multiplier = trial.suggest_float("multiplier", 0.5, 5, step=4.5 / 20)

        # Gather selected features
        selected_features = [
            param_grid_choice_feature[f"feature_{i}_choice"]
            for i in range(self.nb_max_features)
        ]

        # gather points of selected features
        score_vector = [
            param_grid_point[f"feature_{i}_point"] for i in range(self.nb_max_features)
        ]

        # Compute scores for all samples by multiplying selected binary features of X by corresponding points
        score_samples = np.matmul(
            X[selected_features].values, np.transpose(score_vector)
        )

        # Compute associated probabilities
        score_samples_associated_probabilities = self._predict_proba_score(
            score_samples, score_intercept, multiplier
        )

        # Compute logloss
        logloss_samples = log_loss(y, score_samples_associated_probabilities)

        # penalty_0_point = 5 if 0 in score_vector else 0

        return logloss_samples  # * penalty_0_point

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Function that search best parameters (choice of binary features, points and intercept) of a risk score model with Optuna

        This functions calls Optuna to find the best parameters of a risk score model and then construct the feature-point card and score card attributes.

        Args:
            X (pandas.DataFrame): Dataset of features to fit the risk score model on
            y (pandas.Series): Target binary values
        """

        dataframe_schema = pa.DataFrameSchema(
            {c: pa.Column(checks=[pa.Check.isin([0, 1, 0.0, 1.0])]) for c in X.columns},
            strict=True,  # Disable check of other columns in the dataframe
        )

        dataframe_schema.validate(X)
        # Setting the logging level WARNING, the INFO logs are suppressed.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create Optuna study
        self._study = optuna.create_study(
            direction="minimize", study_name="Score search"
        )

        # Define optuna study objective
        optuna_objective = lambda trial: self.score_logloss_objective(
            trial,
            X,
            y,
        )

        self._study.optimize(optuna_objective, **self.optuna_optimize_params)

        # Get best selection of features
        selected_scorepyo_features = [
            self._study.best_params[f"feature_{i}_choice"]
            for i in range(self.nb_max_features)
            if self._study.best_params[f"feature_{i}_point"] != 0
        ]

        # Get best associated points
        selected_scorepyo_features_point = [
            self._study.best_params[f"feature_{i}_point"]
            for i in range(self.nb_max_features)
            if self._study.best_params[f"feature_{i}_point"] != 0
        ]

        # Get best associated intercept
        self._intercept = self._study.best_params["intercept"]

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=selected_scorepyo_features)
        self.feature_point_card[self._POINT_COL] = selected_scorepyo_features_point
        if self._df_info is not None:
            self._df_info = self._df_info.rename({"feature": self._FEATURE_COL}, axis=1)
            self.feature_point_card = self.feature_point_card.merge(
                self._df_info, left_index=True, right_on=["binary_feature"]
            )
        else:
            self.feature_point_card[
                "binary_feature"
            ] = self.feature_point_card.index.values
            self.feature_point_card[
                self._FEATURE_COL
            ] = self.feature_point_card.index.values

        self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
        self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
            "binary_feature"
        ].values

        # Compute score card

        # Minimum score is the sum of all negative points
        min_range_score = sum(
            np.clip(i, a_min=None, a_max=0)
            for i in self.feature_point_card[self._POINT_COL].values
        )

        # Maximum score is the sum of all positive points
        max_range_score = sum(
            np.clip(i, a_min=0, a_max=None)
            for i in self.feature_point_card[self._POINT_COL].values
        )

        # Compute risk score for all integers within min_range_score and max_range_score
        possible_scores = list(range(min_range_score, max_range_score + 1))

        # Compute associated probability of each possible score
        possible_risks = [
            self._predict_proba_score(s, self._intercept) for s in possible_scores
        ]

        # Compute nice display of probabilities
        possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

        # Assign dataframe to score_card attribute
        self.score_card = pd.DataFrame(
            index=["SCORE", "RISK", "_RISK_FLOAT"],
            data=[possible_scores, possible_risks_pct, possible_risks],
        )


class _EBMRiskScoreAuxiliaryLoss(_BaseRiskScore):
    """
    Risk score model based on Optuna.

    This class is a child class of _BaseRiskScore. It implements the fit method that creates the feature-point card and score card attribute.
    It computes them by leveraging the sampling efficiency of Optuna. Optuna is asked to select nb_max_features among all features, and assign points
    to each selected feature. It minimizes the logloss on a given dataset.


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
    fit(self, X, y):
        function creating the feature-point card and score card attributes via Optuna

    score_logloss_objective(self, trial, X, y):
        function that defines the logloss function used by Optuna


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
        optimization_method: Optional[str] = "auxiliary_loss_rounding",
        optimization_options: Optional[dict[str, str]] = None,
        df_info: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            nb_max_features (int): Number of maximum binary features to be selected
            min_point_value (int, optional): Minimum point assigned to a binary feature. Defaults to -2.
            max_point_value (int, optional): Maximum point assigned to a binary feature. Defaults to 3.
            df_info (pandas.DataFrame, optional): DataFrame linking original feature and binary feature. Defaults to None.
            optuna_optimize_params (dict, optional): parameters for optuna optimize function. Defaults to None.
        """
        super().__init__(nb_max_features, min_point_value, max_point_value, df_info)

        # TODO Check value
        self.max_number_binaries_by_features: int = max_number_binaries_by_features
        self._binarizer = AutoBinarizer(
            max_number_binaries_by_features=self.max_number_binaries_by_features
        )
        self.optimization_method = optimization_method
        self.optimization_options = (
            dict() if optimization_options is None else optimization_options
        )
        if "nb_additional_features" not in self.optimization_options.keys():
            self.optimization_options["nb_additional_features"] = 3

        if self.optimization_method == "auxiliary_loss_rounding":
            if "nb_multipliers" not in self.optimization_options.keys():
                self.optimization_options["nb_multipliers"] = 20

    @staticmethod
    def auxiliary_loss_rounding(X, y, w_original, n_multipliers, max_point_value):
        """Auxiliary loss rounding from FasterRisk : https://arxiv.org/pdf/2210.05846.pdf

        Args:
            X (_type_): _description_
            y (_type_): _description_
            w_original (_type_): _description_
            n_multipliers (_type_): _description_
            max_point_value (_type_): _description_

        Returns:
            _type_: _description_
        """
        # max_point_value = self.max_point_value
        max_value_multipliers = max_point_value / np.max(np.abs(w_original))

        min_value_multipliers = 0.5 if max_value_multipliers == 1 else 1
        # n_multipliers = 50  # 20
        list_multipliers = np.linspace(
            min_value_multipliers, max_value_multipliers, n_multipliers
        )

        matrix_w_multiplied = w_original.reshape(-1, 1) * list_multipliers.reshape(
            1, -1
        )
        y_fasterrisk = np.where(y == 1, 1, -1)

        best_log_loss = 1e9
        best_w = None
        best_m = None
        best_intercept = None

        for j in range(n_multipliers):
            X_augmented = np.concatenate(
                [
                    X.values / list_multipliers[j],
                    np.ones((len(X), 1)),
                ],
                axis=1,
            )
            w_multiplied = matrix_w_multiplied[:, j]
            w_multiplied = w_multiplied.reshape(-1, 1)
            J = [
                j1
                for j1, val in enumerate(w_multiplied)
                if np.ceil(val) != np.floor(val)
            ]

            w_multiplied_floor = np.floor(w_multiplied)
            # w_multiplied_ceil = np.ceil(w_multiplied)
            gamma = np.repeat(w_multiplied_floor.T, repeats=len(X), axis=0)

            Z = np.einsum("ij,i->ij", X_augmented, y_fasterrisk)
            gamma = gamma + (Z <= 0)
            gamma = gamma.astype(float)
            xGamma_sum = np.einsum("ij,ij->i", X_augmented, gamma)
            denominator_part = np.einsum("i,i->i", y_fasterrisk, xGamma_sum)
            l = 1 / (1 + np.exp(denominator_part))

            while len(J) > 0:
                U = dict()
                for j1 in J:
                    w_j_up = w_multiplied.copy()
                    w_j_up[j1, :] = np.ceil(w_multiplied[j1, :])
                    lX = np.einsum("i,ij->ij", l, X_augmented)
                    substract_up = (w_j_up - w_multiplied).astype(float)
                    U_j_up = np.sum(np.einsum("ij,jk->ik", lX, substract_up) ** 2)

                    w_j_down = w_multiplied.copy()
                    w_j_down[j1, :] = np.floor(w_multiplied[j1, :])
                    substract_down = (w_j_down - w_multiplied).astype(float)
                    U_j_down = np.sum(np.einsum("ij,jk->ik", lX, substract_down) ** 2)
                    U[j1] = {"up": U_j_up, "down": U_j_down}

                j_up, U_up = min([(j1, U[j1]["up"]) for j1 in J], key=lambda t: t[1])
                j_down, U_down = min(
                    [(j1, U[j1]["down"]) for j1 in J], key=lambda t: t[1]
                )
                if U_up <= U_down:
                    J = [j1 for j1 in J if j1 != j_up]
                    w_multiplied[j_up] = np.ceil(w_multiplied[j_up])
                else:
                    J = [j1 for j1 in J if j1 != j_down]
                    w_multiplied[j_down] = np.floor(w_multiplied[j_down])
                # break

            w_multiplied_features = w_multiplied[:-1]
            w_multiplied_intercept = w_multiplied[-1]
            # w_multiplied_features, w_multiplied_intercept
            XW = np.einsum(
                "ij,jk->ik",
                X_augmented.astype(float),
                w_multiplied.astype(float),
            )
            logloss = np.mean(1 + np.exp(np.einsum("i,ik->i", -y_fasterrisk, XW)))

            if logloss < best_log_loss:
                best_log_loss = logloss
                best_intercept = w_multiplied_intercept
                best_w = w_multiplied_features
                best_m = list_multipliers[j]

        return (
            best_log_loss,
            best_intercept,
            best_w,
            best_m,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_features=None) -> None:
        """Function that search best parameters (choice of binary features, points and intercept) of a risk score model with Optuna

        This functions calls Optuna to find the best parameters of a risk score model and then construct the feature-point card and score card attributes.

        Args:
            X (pandas.DataFrame): Dataset of features to fit the risk score model on
            y (pandas.Series): Target binary values
        """

        # dataframe_schema = pa.DataFrameSchema(
        #     {c: pa.Column(checks=[pa.Check.isin([0, 1, 0.0, 1.0])]) for c in X.columns},
        #     strict=True,  # Disable check of other columns in the dataframe
        # )

        # dataframe_schema.validate(X)

        self._binarizer.fit(X, y, categorical_features=categorical_features)
        df_info = self._binarizer.df_score_feature

        X_binarized = self._binarizer.transform(X)

        df_info["abs_contribution"] = df_info[
            "EBM_log_odds_contribution"
        ].abs() * df_info["density"].fillna(0).astype(int).pow(
            0.95
        )  # to de emphasize large values impact
        nb_max_features = self.nb_max_features
        nb_additional_features = int(
            self.optimization_options["nb_additional_features"]
        )
        n_multipliers = int(self.optimization_options["nb_multipliers"])
        pool_top_features = df_info.sort_values(
            by="abs_contribution", ascending=False
        ).index[: nb_max_features + nb_additional_features]
        best_log_loss = 1e9
        best_w = None
        best_m = None
        best_intercept = None
        print(
            "Nb of combinations:",
            len(list(itertools.combinations(pool_top_features, nb_max_features))),
        )
        i = 0
        for top_features in itertools.combinations(pool_top_features, nb_max_features):
            print("\t", top_features)
            if i % 10 == 0:
                print(i)
            i += 1
            top_features = pd.Index(top_features)
            top_features = top_features.append(pd.Index(["intercept"]))
            df_info_modified = df_info.copy()
            df_info_modified["EBM_log_odds_contribution"] = np.where(
                df_info_modified.index.isin(top_features),
                df_info_modified["EBM_log_odds_contribution"],
                0,
            )
            w_original = df_info_modified["EBM_log_odds_contribution"].values
            (
                logloss,
                w_multiplied_intercept,
                w_multiplied_features,
                multiplier,
            ) = self.auxiliary_loss_rounding(
                X_binarized, y, w_original, n_multipliers, self.max_point_value
            )
            print("\t logloss:", logloss)
            print(
                "\t",
                [
                    (f, w_f)
                    for f, w_f in zip(X_binarized.columns, w_multiplied_features)
                    if w_f != 0
                ],
            )
            print("\t", [w_f for w_f in w_multiplied_features if w_f != 0])
            print("\t intercept:", w_multiplied_intercept)
            print()
            if logloss < best_log_loss:
                best_log_loss = logloss
                best_intercept = w_multiplied_intercept
                best_w = w_multiplied_features
                best_m = multiplier

        self._w = best_w
        self._multiplier = best_m

        # Get best selection of features
        selected_scorepyo_features = [
            f for f, w_f in zip(X_binarized.columns, self._w) if w_f != 0
        ]

        # Get best associated points
        selected_scorepyo_features_point = [w_f[0] for w_f in self._w if w_f != 0]

        # Get best associated intercept
        self._intercept = best_intercept[0]

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=selected_scorepyo_features)
        self.feature_point_card[self._POINT_COL] = selected_scorepyo_features_point
        if self._df_info is not None:
            self._df_info = self._df_info.rename({"feature": self._FEATURE_COL}, axis=1)
            self.feature_point_card = self.feature_point_card.merge(
                self._df_info, left_index=True, right_on=["binary_feature"]
            )
        else:
            self.feature_point_card[
                "binary_feature"
            ] = self.feature_point_card.index.values
            self.feature_point_card[
                self._FEATURE_COL
            ] = self.feature_point_card.index.values

        self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
        self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
            "binary_feature"
        ].values

        # Compute score card

        # Minimum score is the sum of all negative points
        min_range_score = sum(
            np.clip(i, a_min=None, a_max=0)
            for i in self.feature_point_card[self._POINT_COL].values
        )

        # Maximum score is the sum of all positive points
        max_range_score = sum(
            np.clip(i, a_min=0, a_max=None)
            for i in self.feature_point_card[self._POINT_COL].values
        )

        # Compute risk score for all integers within min_range_score and max_range_score
        possible_scores = list(range(min_range_score, max_range_score + 1))

        # Compute associated probability of each possible score
        possible_risks = [
            self._predict_proba_score(s, self._intercept, self._multiplier)
            for s in possible_scores
        ]

        # Compute nice display of probabilities
        possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

        # Assign dataframe to score_card attribute
        self.score_card = pd.DataFrame(
            index=["SCORE", "RISK", "_RISK_FLOAT"],
            data=[possible_scores, possible_risks_pct, possible_risks],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_binarized = self._binarizer.transform(X)
        return super().predict_proba(X_binarized)


class EBMRiskScore(_BaseRiskScore):
    """
    Risk score model based on Optuna.

    This class is a child class of _BaseRiskScore. It implements the fit method that creates the feature-point card and score card attribute.
    It computes them by leveraging the sampling efficiency of Optuna. Optuna is asked to select nb_max_features among all features, and assign points
    to each selected feature. It minimizes the logloss on a given dataset.


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
    fit(self, X, y):
        function creating the feature-point card and score card attributes via Optuna

    score_logloss_objective(self, trial, X, y):
        function that defines the logloss function used by Optuna


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

    DICT_SCORING_METHOD = {
        "average_precision": lambda y_true, y_proba: -average_precision_score(
            y_true, y_proba
        ),
        "log_loss": log_loss,
    }

    def __init__(
        self,
        nb_max_features: int = 4,
        min_point_value: int = -2,
        max_point_value: int = 3,
        max_number_binaries_by_features: int = 3,
        optimization_metric: Optional[str] = "log_loss",
        optimization_options: Optional[dict[str, str]] = None,
        df_info: Optional[pd.DataFrame] = None,
        optuna_optimize_params: Optional[dict] = None,
    ):
        """
        Args:
            nb_max_features (int): Number of maximum binary features to be selected
            min_point_value (int, optional): Minimum point assigned to a binary feature. Defaults to -2.
            max_point_value (int, optional): Maximum point assigned to a binary feature. Defaults to 3.
            df_info (pandas.DataFrame, optional): DataFrame linking original feature and binary feature. Defaults to None.
            optuna_optimize_params (dict, optional): parameters for optuna optimize function. Defaults to None.
        """
        super().__init__(nb_max_features, min_point_value, max_point_value, df_info)

        # TODO Check value
        self.max_number_binaries_by_features: int = max_number_binaries_by_features
        self._binarizer = AutoBinarizer(
            max_number_binaries_by_features=self.max_number_binaries_by_features
        )
        self.optimization_metric = optimization_metric
        self.optimization_options = (
            dict() if optimization_options is None else optimization_options
        )
        if "nb_additional_features" not in self.optimization_options.keys():
            self.optimization_options["nb_additional_features"] = 4

        if "nb_multipliers" not in self.optimization_options.keys():
            self.optimization_options["nb_multipliers"] = 20

        if "min_value_multiplier" not in self.optimization_options.keys():
            self.optimization_options["min_value_multiplier"] = 0.5

        if "max_value_multiplier" not in self.optimization_options.keys():
            self.optimization_options["max_value_multiplier"] = 5

        if optuna_optimize_params is None:
            self.optuna_optimize_params = dict()
            self.optuna_optimize_params["n_trials"] = 200
            self.optuna_optimize_params["timeout"] = 15
        else:
            self.optuna_optimize_params = optuna_optimize_params

    def score_objective(
        self, trial, X: pd.DataFrame, y: pd.Series, dict_bounds, metric
    ) -> float:
        if dict_bounds is None:
            dict_bounds = {
                f: {
                    "lower_bound": self.min_point_value,
                    "upper_bound": self.max_point_value,
                }
                for f in X.columns
            }
        else:
            for f in X.columns:
                if f not in dict_bounds.keys():
                    dict_bounds[f] = {
                        "lower_bound": self.min_point_value,
                        "upper_bound": self.max_point_value,
                    }
        param_grid_point = {
            f"feature_{i}_point": trial.suggest_int(
                f"feature_{i}_point",
                dict_bounds[i]["lower_bound"],
                dict_bounds[i]["upper_bound"],
            )
            for i in X.columns
        }

        # Define the parameter of trial for the intercept of the risk score model
        score_intercept = trial.suggest_float(
            "intercept",
            -10 + self.min_point_value * self.nb_max_features,
            10 + self.max_point_value * self.nb_max_features,
            step=0.5,
        )

        # Define the parameter of trial for the intercept of the risk score model
        multiplier = trial.suggest_float(
            "multiplier",
            self.optimization_options["min_value_multiplier"],
            self.optimization_options["max_value_multiplier"],
            step=(
                self.optimization_options["max_value_multiplier"]
                - self.optimization_options["min_value_multiplier"]
            )
            / self.optimization_options["nb_multipliers"],
        )

        # gather points of selected features
        score_vector = [param_grid_point[f"feature_{i}_point"] for i in X.columns]

        # Compute scores for all samples by multiplying selected binary features of X by corresponding points
        score_samples = np.matmul(X.values, np.transpose(score_vector))

        # Compute associated probabilities
        score_samples_associated_probabilities = self._predict_proba_score(
            score_samples, score_intercept, multiplier
        )

        # Compute logloss
        logloss_samples = log_loss(y, score_samples_associated_probabilities)

        average_precision_samples = average_precision_score(
            y.astype(int), score_samples_associated_probabilities
        )
        chosen_metric_samples = metric(
            y.astype(int), score_samples_associated_probabilities
        )
        # chosen_metric_samples = self.DICT_SCORING_METHOD[self.optimization_metric](
        #     y.astype(int), score_samples_associated_probabilities
        # )

        # penalty_0_point = 5 if 0 in score_vector else 0
        return chosen_metric_samples
        # return average_precision_samples, logloss_samples
        # return logloss_samples  # * penalty_0_point

    def score_objective_proba(
        self, trial, X: pd.DataFrame, y: pd.Series, dict_bounds
    ) -> float:
        if dict_bounds is None:
            dict_bounds = {
                f: {
                    "lower_bound": self.min_point_value,
                    "upper_bound": self.max_point_value,
                }
                for f in X.columns
            }
        else:
            for f in X.columns:
                if f not in dict_bounds.keys():
                    dict_bounds[f] = {
                        "lower_bound": self.min_point_value,
                        "upper_bound": self.max_point_value,
                    }
        param_grid_point = {
            f"feature_{i}_point": trial.suggest_int(
                f"feature_{i}_point",
                dict_bounds[i]["lower_bound"],
                dict_bounds[i]["upper_bound"],
            )
            for i in X.columns
        }

        # # Define the parameter of trial for the intercept of the risk score model
        # score_intercept = trial.suggest_float(
        #     "intercept",
        #     -10 + self.min_point_value * self.nb_max_features,
        #     10 + self.max_point_value * self.nb_max_features,
        #     step=0.5,
        # )

        # # Define the parameter of trial for the intercept of the risk score model
        # multiplier = trial.suggest_float(
        #     "multiplier",
        #     self.optimization_options["min_value_multiplier"],
        #     self.optimization_options["max_value_multiplier"],
        #     step=(
        #         self.optimization_options["max_value_multiplier"]
        #         - self.optimization_options["min_value_multiplier"]
        #     )
        #     / self.optimization_options["nb_multipliers"],
        # )

        # gather points of selected features
        score_vector = [param_grid_point[f"feature_{i}_point"] for i in X.columns]

        # Compute scores for all samples by multiplying selected binary features of X by corresponding points
        score_samples = np.matmul(X.values, np.transpose(score_vector))

        # Compute associated probabilities
        score_samples_associated_probabilities = (
            score_samples - np.min(score_samples)
        ) / (np.max(score_samples) - np.min(score_samples) + 1e9)

        average_precision_samples = average_precision_score(
            y.astype(int), score_samples_associated_probabilities
        )
        return average_precision_samples

    def score_objective_feature_points(
        self, trial, X: pd.DataFrame, y: pd.Series, dict_bounds
    ) -> float:
        if dict_bounds is None:
            dict_bounds = {
                f: {
                    "lower_bound": self.min_point_value,
                    "upper_bound": self.max_point_value,
                }
                for f in X.columns
            }
        else:
            for f in X.columns:
                if f not in dict_bounds.keys():
                    dict_bounds[f] = {
                        "lower_bound": self.min_point_value,
                        "upper_bound": self.max_point_value,
                    }
        param_grid_point = {
            f"feature_{i}_point": trial.suggest_int(
                f"feature_{i}_point",
                dict_bounds[i]["lower_bound"],
                dict_bounds[i]["upper_bound"],
            )
            for i in X.columns
        }

        # # Define the parameter of trial for the intercept of the risk score model
        # score_intercept = trial.suggest_float(
        #     "intercept",
        #     -10 + self.min_point_value * self.nb_max_features,
        #     10 + self.max_point_value * self.nb_max_features,
        #     step=0.5,
        # )

        # # Define the parameter of trial for the intercept of the risk score model
        # multiplier = trial.suggest_float(
        #     "multiplier",
        #     self.optimization_options["min_value_multiplier"],
        #     self.optimization_options["max_value_multiplier"],
        #     step=(
        #         self.optimization_options["max_value_multiplier"]
        #         - self.optimization_options["min_value_multiplier"]
        #     )
        #     / self.optimization_options["nb_multipliers"],
        # )

        # gather points of selected features
        score_vector = [param_grid_point[f"feature_{i}_point"] for i in X.columns]

        # Compute scores for all samples by multiplying selected binary features of X by corresponding points
        score_samples = np.matmul(X.values, np.transpose(score_vector))

        # Compute associated probabilities
        score_samples_associated_probabilities = self._predict_proba_score(
            score_samples, score_intercept, multiplier
        )

        # Compute logloss
        average_precision_samples = log_loss(y, score_samples_associated_probabilities)
        return average_precision_samples

    def optuna_optim(self, X, y, metric, dict_bounds=None):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create Optuna study
        study_feature_points = optuna.create_study(
            # directions=["minimize", "minimize"],
            direction="minimize",
            study_name="Score search",
        )

        # Define optuna study objective
        optuna_objective = lambda trial: self.score_objective_feature_points(
            trial, X, y, dict_bounds
        )

        study_feature_points.optimize(optuna_objective, **self.optuna_optimize_params)

        # trial_with_highest_average_precision = max(
        #     study.best_trials, key=lambda t: t.values[0]
        # )

        # Get best associated points
        selected_scorepyo_features_point = [
            study_feature_points.best_params[f"feature_{i}_point"]
            # trial_with_highest_average_precision.params[f"feature_{i}_point"]
            for i in X.columns
            # if study.best_params[f"feature_{i}_point"] != 0
        ]

        best_average_precision = study.study_feature_points
        # best_logloss = trial_with_highest_average_precision.values[0]

        # Get best associated intercept
        # intercept = study.best_params["intercept"]
        # intercept = trial_with_highest_average_precision.params["intercept"]

        # multiplier = study.best_params["multiplier"]
        # multiplier = trial_with_highest_average_precision.params["multiplier"]

        return best_average_precision, selected_scorepyo_features_point

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_features=None) -> None:
        """Function that search best parameters (choice of binary features, points and intercept) of a risk score model with Optuna

        This functions calls Optuna to find the best parameters of a risk score model and then construct the feature-point card and score card attributes.

        Args:
            X (pandas.DataFrame): Dataset of features to fit the risk score model on
            y (pandas.Series): Target binary values
        """

        # Binarize the features with the AutoBinarizer class
        self._binarizer.fit(X, y, categorical_features=categorical_features)
        df_info = self._binarizer.df_score_feature

        X_binarized = self._binarizer.transform(X)

        # Rank the binary feature by likeliness to be important for the risk score model
        # The current estimated importance is the log odd computed by the EBM model x number of positive samples for
        # that binary feature.
        # The cardinality of the positive samples is de emphasize by taking the 0.95 power.
        # TODO: take into account the impact of having mixed class in the samples

        df_info["abs_contribution"] = df_info[
            "EBM_log_odds_contribution"
        ].abs() * df_info["density"].fillna(0).astype(int).pow(
            0.95
        )  # to de emphasize large values impact
        nb_max_features = self.nb_max_features
        nb_additional_features = int(
            self.optimization_options["nb_additional_features"]
        )

        # Compute the reduced pool of top features to choose from
        pool_top_features = df_info.sort_values(
            by="abs_contribution", ascending=False
        ).index[: nb_max_features + nb_additional_features]

        # Compute bounds for points for each feature to reduce optimization space
        # -> negative points for negative log odds
        # -> positive points for positive log odds
        df_sense = df_info.sort_values(by="abs_contribution", ascending=False).iloc[
            : nb_max_features + nb_additional_features
        ]
        df_sense["lower_bound_point"] = np.where(
            df_sense["EBM_log_odds_contribution"] > 0,
            min([1, self.max_point_value]),
            self.min_point_value,
        )
        df_sense["upper_bound_point"] = np.where(
            df_sense["EBM_log_odds_contribution"] > 0,
            self.max_point_value,
            max([-1, self.min_point_value]),
        )

        best_metric = 1e9
        best_m = None
        best_intercept = None
        best_features_point = None
        best_features = None

        # For each combination of nb_max_features within the pool of selected top features
        # compute an optuna optimized risk score model and keep the one maximizing the chosen metric
        for top_features in itertools.combinations(pool_top_features, nb_max_features):
            top_features = list(top_features)

            # For each feature, compute lower and upper bound for point value (ie negative or positive points)
            dict_bounds = {
                f: {
                    "lower_bound": df_sense.loc[f, "lower_bound_point"],
                    "upper_bound": df_sense.loc[f, "upper_bound_point"],
                }
                for f in top_features
            }
            (
                current_metric,
                selected_scorepyo_features_point,
                intercept,
                multiplier,
            ) = self.optuna_optim(
                X_binarized[top_features],
                y,
                self.DICT_SCORING_METHOD["average_precision"],
                dict_bounds,
            )

            if current_metric < best_metric:
                best_metric = current_metric
                best_intercept = intercept
                best_features_point = selected_scorepyo_features_point
                best_features = top_features
                best_m = multiplier

        top_features = list(best_features)

        # For each feature, compute lower and upper bound for point value (ie negative or positive points)
        dict_bounds = {
            f: {
                "lower_bound": df_sense.loc[f, "lower_bound_point"],
                "upper_bound": df_sense.loc[f, "upper_bound_point"],
            }
            for f in top_features
        }
        (
            current_metric,
            selected_scorepyo_features_point,
            intercept,
            multiplier,
        ) = self.optuna_optim(
            X_binarized[top_features],
            y,
            self.DICT_SCORING_METHOD["log_loss"],
            dict_bounds,
        )
        best_metric = 1e9  # TODO put current logloss of best run

        if current_metric < best_metric:
            best_metric = current_metric
            best_intercept = intercept
            best_features_point = selected_scorepyo_features_point
            best_features = top_features
            best_m = multiplier

        self._multiplier = best_m

        # Get best selection of features
        selected_scorepyo_features = [
            f for f, w_f in zip(best_features, best_features_point) if w_f != 0
        ]

        # Get best associated points
        selected_scorepyo_features_point = [
            w_f for w_f in best_features_point if w_f != 0
        ]

        # Get best associated intercept
        self._intercept = best_intercept

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=selected_scorepyo_features)
        self.feature_point_card[self._POINT_COL] = selected_scorepyo_features_point
        if self._df_info is not None:
            self._df_info = self._df_info.rename({"feature": self._FEATURE_COL}, axis=1)
            self.feature_point_card = self.feature_point_card.merge(
                self._df_info, left_index=True, right_on=["binary_feature"]
            )
        else:
            self.feature_point_card[
                "binary_feature"
            ] = self.feature_point_card.index.values
            self.feature_point_card[
                self._FEATURE_COL
            ] = self.feature_point_card.index.values

        self.feature_point_card = self.feature_point_card.set_index(self._FEATURE_COL)
        self.feature_point_card[self._DESCRIPTION_COL] = self.feature_point_card[
            "binary_feature"
        ].values

        # Compute score card

        # Minimum score is the sum of all negative points
        min_range_score = sum(
            np.clip(i, a_min=None, a_max=0)
            for i in self.feature_point_card[self._POINT_COL].values
        )

        # Maximum score is the sum of all positive points
        max_range_score = sum(
            np.clip(i, a_min=0, a_max=None)
            for i in self.feature_point_card[self._POINT_COL].values
        )

        # Compute risk score for all integers within min_range_score and max_range_score
        possible_scores = list(range(min_range_score, max_range_score + 1))

        # Compute associated probability of each possible score
        possible_risks = [
            self._predict_proba_score(s, self._intercept, self._multiplier)
            for s in possible_scores
        ]

        # Compute nice display of probabilities
        possible_risks_pct = [f"{r:.2%}" for r in possible_risks]

        # Assign dataframe to score_card attribute
        self.score_card = pd.DataFrame(
            index=["SCORE", "RISK", "_RISK_FLOAT"],
            data=[possible_scores, possible_risks_pct, possible_risks],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_binarized = self._binarizer.transform(X)
        return super().predict_proba(X_binarized)


class EBMRiskScoreNew(_BaseRiskScore):
    """


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
    fit(self, X, y):
        function creating the feature-point card and score card attributes via Optuna

    score_logloss_objective(self, trial, X, y):
        function that defines the logloss function used by Optuna


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

    DICT_SCORING_METHOD = {
        "average_precision": lambda y_true, y_proba: -average_precision_score(
            y_true, y_proba
        ),
        "log_loss": log_loss,
        "roc_auc": lambda y_true, y_proba: -roc_auc_score(y_true, y_proba),
    }

    def __init__(
        self,
        nb_max_features: int = 4,
        min_point_value: int = -2,
        max_point_value: int = 3,
        max_number_binaries_by_features: int = 3,
        optimization_metric: Optional[str] = "log_loss",
        optimization_options: Optional[dict[str, str]] = None,
        nb_additional_features: Optional[int] = 4,
        df_info: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            nb_max_features (int): Number of maximum binary features to be selected
            min_point_value (int, optional): Minimum point assigned to a binary feature. Defaults to -2.
            max_point_value (int, optional): Maximum point assigned to a binary feature. Defaults to 3.
            df_info (pandas.DataFrame, optional): DataFrame linking original feature and binary feature. Defaults to None.
            optuna_optimize_params (dict, optional): parameters for optuna optimize function. Defaults to None.
        """
        super().__init__(nb_max_features, min_point_value, max_point_value, df_info)

        # TODO Check value
        self.max_number_binaries_by_features: int = max_number_binaries_by_features
        self._binarizer = AutoBinarizer(
            max_number_binaries_by_features=self.max_number_binaries_by_features
        )
        self.optimization_metric = optimization_metric
        self._primary_metric = self.DICT_SCORING_METHOD[self.optimization_metric]

        self.nb_additional_features = int(nb_additional_features)
        self.optimization_options = (
            dict() if optimization_options is None else optimization_options
        )
        if "nb_additional_features" not in self.optimization_options.keys():
            self.optimization_options["nb_additional_features"] = 4

    @staticmethod
    def _compute_ranking_metric_scenario(score_samples_all_scenario, y, ranking_metric):
        # Compute the minimum and maximum score for each scenario
        # This will be used to generate easy first set of probabilities for ranking metrics (average precision and roc auc)
        min_score_samples_all_scenario = np.min(score_samples_all_scenario, axis=0)
        max_score_samples_all_scenario = np.max(score_samples_all_scenario, axis=0)

        # Transform score into dummy probabilities but well ranked
        # Done with a min-max rescale
        score_samples_associated_probabilities = (
            score_samples_all_scenario - min_score_samples_all_scenario
        ) / (max_score_samples_all_scenario - min_score_samples_all_scenario)
        score_samples_associated_probabilities.shape

        # Compute metric on probabilities
        # TODO make it generic to metric (logloss, AP, roc auc)
        metric_scores = [
            ranking_metric(y, score_samples_associated_probabilities[:, j])
            for j in range(score_samples_associated_probabilities.shape[-1])
        ]

        return metric_scores

    @staticmethod
    def _compute_logloss_metric_scenario(
        best_points_scenario, X_binarized, y, top_features
    ):

        # best_points_scenario = all_points_possibilities[idx_best_scenario, :]

        df_calibration = pd.DataFrame(columns=["target", "score"])
        df_calibration["target"] = y.values
        df_calibration["score"] = np.matmul(
            X_binarized[list(top_features).copy()].values, best_points_scenario
        )

        df_score_proba_association = df_calibration.groupby("score")["target"].mean()
        df_score_proba_association.columns = ["proba"]
        min_sum_point = np.sum(np.clip(best_points_scenario, a_max=0, a_min=None))
        max_sum_point = np.sum(np.clip(best_points_scenario, a_min=0, a_max=None))
        full_index = np.arange(min_sum_point, max_sum_point + 1e-3)
        missing_index = set(full_index) - set(df_score_proba_association.index)

        df_score_proba_association = df_score_proba_association.reindex(full_index)
        df_score_proba_association = df_score_proba_association.interpolate(
            method="linear"
        )

        df_reordering = pd.DataFrame(df_score_proba_association.copy())
        #     # print(df_reordering)
        # Count cardinality of each score
        df_reordering["count"] = df_calibration.groupby("score")["target"].count()
        df_reordering["count"] = df_reordering["count"].fillna(0)

        df_cvx = df_reordering.copy()
        # import cvxpy as cp

        positive_sample_count = (
            df_cvx["target"].values * df_cvx["count"].values
        ).astype(int)
        negative_sample_count = df_cvx["count"].values - positive_sample_count

        list_proba = [cp.Variable(1) for _ in df_cvx.index]

        total_count = df_cvx["count"].sum()

        list_expression = [
            -cp.log(p) * w_pos - cp.log(1 - p) * w_neg
            for p, w_pos, w_neg in zip(
                list_proba, positive_sample_count, negative_sample_count
            )
        ]
        objective = cp.Minimize(cp.sum(list_expression) / total_count)  # Objective
        constraints = []
        for p in list_proba:
            constraints.append(p >= 0)
            constraints.append(p <= 1)
        for i in range(1, len(list_proba)):
            constraints.append(list_proba[i] - list_proba[i - 1] - 1e-3 >= 0)

        problem = cp.Problem(objective, constraints)
        try:
            opt = problem.solve()
        except:
            opt = 1e6
        return opt

    @staticmethod
    def _compute_logloss_all_scenario(
        all_points_possibilities, X_binarized, y, top_features
    ):

        best_logloss = 1e6
        for idx_scenario in range(all_points_possibilities.shape[0]):
            best_points_scenario = all_points_possibilities[idx_scenario, :]
            opt = EBMRiskScoreNew._compute_logloss_metric_scenario(
                best_points_scenario, X_binarized, y, top_features
            )

            if opt < best_logloss:
                best_logloss = opt
                idx_best_scenario = idx_scenario

        return idx_best_scenario, best_logloss

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_calib: pd.DataFrame = None,
        y_calib: pd.Series = None,
        categorical_features=None,
    ):

        # Binarize the features with the AutoBinarizer class
        self._binarizer.fit(X, y, categorical_features=categorical_features)
        df_info = self._binarizer.df_score_feature

        X_binarized = self._binarizer.transform(X)

        # Rank the binary feature by likeliness to be important for the risk score model
        # The current estimated importance is the log odd computed by the EBM model x number of positive samples for
        # that binary feature.
        # The cardinality of the positive samples is de emphasize by taking the 0.95 power.
        # TODO: take into account the impact of having mixed class in the samples

        df_info["abs_contribution"] = df_info[
            "EBM_log_odds_contribution"
        ].abs() * df_info["density"].fillna(0).astype(int).pow(
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
            df_sense["EBM_log_odds_contribution"] > 0,
            min([1, self.max_point_value]),
            self.min_point_value,
        )
        df_sense["upper_bound_point"] = np.where(
            df_sense["EBM_log_odds_contribution"] > 0,
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
        chosen_set_fasterrisk = set(
            [
                "the slope of the peak exercise ST segment < 1.5",
                "number of major vessels (0-3) colored by flourosopy < 0.5",
                "chest pain type_4.0",
                "thal_7.0",
            ]
        )
        _count = 0
        # For all combinations of nb_max_feature from the set of selected binary features
        nb_combi = len(
            list(itertools.combinations(pool_top_features, self.nb_max_features))
        )
        for top_features in itertools.combinations(
            pool_top_features, self.nb_max_features
        ):
            tagged = len(chosen_set_fasterrisk - set(list(top_features))) == 0
            if _count % 10 == 0:
                print(_count / nb_combi)
            _count += 1
            # Gather all point ranges for each binary feature
            all_points_by_feature = [
                dict_point_ranges[f]["all_points"] for f in top_features
            ]

            # Compute the cartesian product of all possible point values for each feature
            # This creates a nxd matrix with n being the number of combinations of points for each binary feature
            # and d being the number of selected binary features
            all_points_possibilities = np.array(
                list(itertools.product(*all_points_by_feature))
            )

            # TODO use np.argpartition in an evolution to take best logloss among best AP or other
            # or conversely by giving a secondary metric and a size of contestant on secondary metric
            if self.optimization_metric == "log_loss":
                (
                    idx_best_scenario,
                    best_logloss_scenario,
                ) = self._compute_logloss_all_scenario(
                    all_points_possibilities, X_binarized, y, top_features
                )
                if best_metric > best_logloss_scenario:
                    best_metric = best_logloss_scenario
                    best_scenario = all_points_possibilities[
                        idx_best_scenario, :
                    ].copy()
                    best_feature_selection = list(top_features).copy()

            else:
                # Compute the score of samples in each scenario for all samples
                score_samples_all_scenario = np.matmul(
                    X_binarized[list(top_features)].values, all_points_possibilities.T
                )

                average_precision_scores = self._compute_ranking_metric_scenario(
                    score_samples_all_scenario, y, average_precision_score
                )

                # TODO use np.argpartition in an evolution to take best logloss among best AP or other
                # or conversely by giving a secondary metric and a size of contestant on secondary metric
                # size_pool_scenario = 10
                # ind_top_scenario = np.argpartition(
                #     average_precision_scores, -size_pool_scenario
                # )[-size_pool_scenario:]
                average_precision_best_scenario = -np.max(average_precision_scores)

                size_pool_scenario = 10
                ind_top_scenario = np.argpartition(
                    average_precision_scores, -size_pool_scenario
                )[-size_pool_scenario:]

                if tagged:
                    for ind in ind_top_scenario:
                        print(average_precision_scores[ind])
                        print(all_points_possibilities[ind, :])
                        print()
                (
                    idx_best_scenario,
                    best_logloss_scenario,
                ) = self._compute_logloss_all_scenario(
                    all_points_possibilities[ind_top_scenario, :],
                    X_binarized,
                    y,
                    top_features,
                )
                if tagged:
                    print(average_precision_scores[ind_top_scenario[idx_best_scenario]])
                    print(all_points_possibilities[ind_top_scenario[idx_best_scenario], :])
                    print(best_logloss_scenario)
                    print()
                    print()

                if best_metric > best_logloss_scenario:
                    best_metric = best_logloss_scenario
                    idx_best_scenario = ind_top_scenario[idx_best_scenario]
                    best_scenario = all_points_possibilities[
                        idx_best_scenario, :
                    ].copy()
                    best_feature_selection = list(top_features).copy()

                ##### Previous version ####
                # idx_best_scenario = np.argmax(average_precision_scores)
                # best_points_scenario = all_points_possibilities[idx_best_scenario, :]

                # if best_metric > average_precision_best_scenario:
                #     best_metric = average_precision_best_scenario
                #     best_scenario = best_points_scenario.copy()
                #     best_feature_selection = list(top_features).copy()

        df_calibration = pd.DataFrame(columns=["target", "score"])
        df_calibration["target"] = y.values
        df_calibration["score"] = np.matmul(
            X_binarized[best_feature_selection].values, best_scenario
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
        # print(df_reordering)
        # Count cardinality of each score
        df_reordering["count"] = df_calibration.groupby("score")["target"].count()
        df_reordering["count"] = df_reordering["count"].fillna(0)

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

        opt = problem.solve()

        # Get the optimized value for the probabilities
        df_reordering["sorted_proba"] = [p.value[0] for p in list_proba]

        # print(df_reordering)
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

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_binarized = self._binarizer.transform(X)
        return super().predict_proba(X_binarized)
