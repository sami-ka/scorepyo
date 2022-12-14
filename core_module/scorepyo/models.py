"""Classes to create and fit risk-score type model.
"""
import itertools
import numbers
from abc import abstractmethod
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import pandera as pa
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator

# from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss

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
            score_samples, score_intercept
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

    def __init__(
        self,
        nb_max_features: int = 4,
        min_point_value: int = -2,
        max_point_value: int = 3,
        max_number_binaries_by_features: int = 3,
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

        X_binarized = self._binarizer.transform_bis(X)

        df_info["abs_contribution"] = (
            df_info["EBM_log_odds_contribution"].abs() * df_info["density"]
        )
        nb_max_features = self.nb_max_features
        nb_additional_features = 11
        pool_top_features = df_info.sort_values(
            by="abs_contribution", ascending=False
        ).index[: nb_max_features + nb_additional_features]
        best_log_loss = 1e9
        best_w = None
        best_m = None
        best_intercept = None
        for top_features in itertools.combinations(pool_top_features, nb_max_features):
            # top_features = df_info.sort_values(
            #     by="abs_contribution", ascending=False
            # ).index[:nb_max_features]
            top_features = pd.Index(top_features)
            top_features = top_features.append(pd.Index(["intercept"]))
            df_info_modified = df_info.copy()
            df_info_modified["EBM_log_odds_contribution"] = np.where(
                df_info_modified.index.isin(top_features),
                df_info_modified["EBM_log_odds_contribution"],
                0,
            )

            max_point_value = self.max_point_value
            max_value_multipliers = (
                max_point_value
                / df_info_modified["EBM_log_odds_contribution"].fillna(0).abs().max()
            )

            min_value_multipliers = 0.5 if max_value_multipliers == 1 else 1
            n_multipliers = 50  # 20
            list_multipliers = np.linspace(
                min_value_multipliers, max_value_multipliers, n_multipliers
            )

            w_original = df_info_modified["EBM_log_odds_contribution"].values

            matrix_w_multiplied = w_original.reshape(-1, 1) * list_multipliers.reshape(
                1, -1
            )
            y_fasterrisk = np.where(y == 1, 1, -1)
            # X_train_binarized_augmented = np.concatenate([X_train_binarized.values,np.ones((len(X_train_binarized),1))], axis=1)

            for j in range(n_multipliers):
                X_binarized_augmented = np.concatenate(
                    [
                        X_binarized.values / list_multipliers[j],
                        np.ones((len(X_binarized), 1)),
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

                Z = np.einsum("ij,i->ij", X_binarized_augmented, y_fasterrisk)
                gamma = gamma + (Z <= 0)
                gamma = gamma.astype(float)
                xGamma_sum = np.einsum("ij,ij->i", X_binarized_augmented, gamma)
                denominator_part = np.einsum("i,i->i", y_fasterrisk, xGamma_sum)
                l = 1 / (1 + np.exp(denominator_part))

                while len(J) > 0:
                    U = dict()
                    for j1 in J:
                        w_j_up = w_multiplied.copy()
                        w_j_up[j1, :] = np.ceil(w_multiplied[j1, :])
                        lX = np.einsum("i,ij->ij", l, X_binarized_augmented)
                        substract_up = (w_j_up - w_multiplied).astype(float)
                        U_j_up = np.sum(np.einsum("ij,jk->ik", lX, substract_up) ** 2)

                        w_j_down = w_multiplied.copy()
                        w_j_down[j1, :] = np.floor(w_multiplied[j1, :])
                        substract_down = (w_j_down - w_multiplied).astype(float)
                        U_j_down = np.sum(
                            np.einsum("ij,jk->ik", lX, substract_down) ** 2
                        )
                        U[j1] = {"up": U_j_up, "down": U_j_down}

                    j_up, U_up = min(
                        [(j1, U[j1]["up"]) for j1 in J], key=lambda t: t[1]
                    )
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
                    X_binarized_augmented.astype(float),
                    w_multiplied.astype(float),
                )
                logloss = np.mean(1 + np.exp(np.einsum("i,ik->i", -y_fasterrisk, XW)))
                if logloss < best_log_loss:
                    best_log_loss = logloss
                    best_intercept = w_multiplied_intercept
                    best_w = w_multiplied_features
                    best_m = list_multipliers[j]
                    print("\t", best_log_loss)

        self._w = best_w
        self._multiplier = best_m

        # Get best selection of features
        selected_scorepyo_features = [
            f for f, w_f in zip(X_binarized.columns, self._w) if w_f != 0
        ]

        # Get best associated points
        selected_scorepyo_features_point = [w_f[0] for w_f in self._w if w_f != 0]

        # # Get best selection of features
        # selected_scorepyo_features = [
        #     self._study.best_params[f"feature_{i}_choice"]
        #     for i in range(self.nb_max_features)
        #     if self._study.best_params[f"feature_{i}_point"] != 0
        # ]

        # Get best associated points
        # selected_scorepyo_features_point = [
        #     self._study.best_params[f"feature_{i}_point"]
        #     for i in range(self.nb_max_features)
        #     if self._study.best_params[f"feature_{i}_point"] != 0
        # ]

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
        X_binarized = self._binarizer.transform_bis(X)
        return super().predict_proba(X_binarized)
