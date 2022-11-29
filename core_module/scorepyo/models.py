"""TODO Add docstring Basescorecard+Optuna scorecard
TODO Add typing
TODO Comment code
TODO Improve naming and code clarity
TODO add Neurips risk scorer
TODO add class constrained OptunaScorecard
    
"""
import numbers
from abc import abstractmethod

import numpy as np
import optuna
import pandas as pd
import pandera as pa
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss

from scorepyo.exceptions import (
    MinPointOverMaxPointError,
    NegativeValueError,
    NonIntegerValueError,
)


class _BaseScoreCard:
    """
    Base class for score card type model.

    This class provides common functions for score card type model, no matter the way points and binary features are designed.
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

    def __init__(
        self, nb_max_features=4, min_point_value=-2, max_point_value=3, df_info=None
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
        self.nb_max_features = nb_max_features

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
        self.min_point_value = min_point_value
        self.max_point_value = max_point_value

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

        self.feature_point_card = None
        self.score_card = None

    @staticmethod
    def _predict_proba_score(score, intercept):
        """Function that computes the logistic function value at score+intercept value

        Args:
            score (np.array(int)): sum of points coming from binary features
            intercept (float): intercept of score card. log-odds of having 0 point

        Returns:
            np.array(int): associated probability
        """
        score_associated_proba = 1 / (1 + np.exp(-(score + intercept)))
        return score_associated_proba

    @abstractmethod
    def fit(self, X, y):
        """Functions that creates the feature-point card and score card

        Must be defined for each child class

        Args:
            X (pandas.DataFrame): binary feature dataset
            y (pandas.Series): binary target

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def predict(self, X, proba_threshold=0.5):
        """Predicts binary class based on probabillity threshold

        Afer computing the probability for each sample,

        Args:
            X (_type_): _description_
            proba_threshold (float, optional): _description_. Defaults to 0.5.

        Returns:
            nbarray of shape (n_samples,): predicted class based on predicted probability and threshold
        """
        if (self.feature_point_card is None) or (self.score_card is None):
            raise NotFittedError("ScoreCard model has not been fitted yet")
        proba = self.predict_proba(X)

        return (proba[:, 1] >= proba_threshold).astype(int)

    def predict_proba(self, X):
        """Function that outputs probability of positive class according to score card

        Args:
            X (pandas.DataFrame): dataset of features

        Returns:
            ndarray of shape (n_samples, 2): probability of negative and positive class in each column resp.
        """
        if (self.feature_point_card is None) or (self.score_card is None):
            raise NotFittedError("ScoreCard model has not been fitted yet")

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

        points = self.feature_point_card["point"].values
        X_total_points = np.matmul(X_selected_features.values, points)
        proba = df_score_card.loc[X_total_points, "_RISK_FLOAT"].values.reshape(-1, 1)

        return np.concatenate([1 - proba, proba], axis=1)

    def summary(self):
        if (self.feature_point_card is None) or (self.score_card is None):
            raise NotFittedError("ScoreCard model has not been fitted yet")
        print("======================")
        print("| FEATURE-POINT CARD |")
        print("======================")
        print(
            self.feature_point_card[["Description", "point"]]
            .sort_values(by="point")
            .to_markdown()
        )
        print()
        print("=======================================")
        print("=======================================")
        print()
        print("======================")
        print("|     SCORE CARD     |")
        print("======================")
        print(self.score_card.loc[["SCORE", "RISK"], :].to_markdown(headers="firstrow"))


class OptunaScoreCard(_BaseScoreCard):
    """
    Score card model based on Optuna.

    This class is a child class of _BaseScorecard. It implements the fit method that creates the feature-point card and score card attribute.
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


    From _BaseScoreCard:

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
        nb_max_features=4,
        min_point_value=-2,
        max_point_value=3,
        df_info=None,
        optuna_optimize_params=None,
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

    def score_logloss_objective(self, trial, X, y):
        """Logloss objective function for score card exploration parameters sampled with optuna.

        This function creates 2x`self.nb_max_features`+1 parameters for the optuna trial:
        - `self.nb_max_features` categorical parameters for the choice of binary features to build the score card on
        - `self.nb_max_features` integer parameters for the choice of points associated to the selected binary feature
        - one float parameter for the intercept of the score card (i.e. the log odd associated to a score of 0)


        Args:
            trial (Optune.trial): Trial for optuna
            X (pandas.DataFrame): dataset of features to minimize scorecard logloss on
            y (nd.array): Target binary values

        Returns:
            float: log-loss value for score card sampled parameters
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

        # Define the parameters of trial for the points associated to each selected binary feature
        param_grid_point = {
            f"feature_{i}_point": trial.suggest_int(
                f"feature_{i}_point", self.min_point_value, self.max_point_value
            )
            for i in range(self.nb_max_features)
        }

        # Define the parameter of trial for the intercept of the score card
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

        return logloss_samples

    def fit(self, X, y):
        """Function that search best parameters (choice of binary features, points and intercept) of a score card with Optuna

        This functions calls Optuna to find the best parameters of a score card and then construct the feature-point card and score card attributes.

        Args:
            X (pandas.DataFrame): Dataset of features to fit the score card on
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
        study = optuna.create_study(direction="minimize", study_name="Score search")

        # Define optuna study objective
        optuna_objective = lambda trial: self.score_logloss_objective(
            trial,
            X,
            y,
        )

        study.optimize(optuna_objective, **self.optuna_optimize_params)

        # Get best selection of features
        selected_scorepyo_features = [
            study.best_params[f"feature_{i}_choice"]
            for i in range(self.nb_max_features)
            if study.best_params[f"feature_{i}_point"] != 0
        ]

        # Get best associated points
        selected_scorepyo_features_point = [
            study.best_params[f"feature_{i}_point"]
            for i in range(self.nb_max_features)
            if study.best_params[f"feature_{i}_point"] != 0
        ]

        # Get best associated intercept
        self._intercept = study.best_params["intercept"]

        # Build feature-point card
        self.feature_point_card = pd.DataFrame(index=selected_scorepyo_features)
        self.feature_point_card["point"] = selected_scorepyo_features_point
        if self._df_info is not None:
            self.feature_point_card = self.feature_point_card.merge(
                self._df_info, left_index=True, right_on=["binary_feature"]
            )
        else:
            self.feature_point_card[
                "binary_feature"
            ] = self.feature_point_card.index.values
            self.feature_point_card["feature"] = self.feature_point_card.index.values

        self.feature_point_card = self.feature_point_card.set_index("feature")
        self.feature_point_card["Description"] = self.feature_point_card[
            "binary_feature"
        ].values

        # Compute score card

        # Minimum score is the sum of all negative points
        min_range_score = sum(
            np.clip(i, a_min=None, a_max=0)
            for i in self.feature_point_card["point"].values
        )

        # Maximum score is the sum of all positive points
        max_range_score = sum(
            np.clip(i, a_min=0, a_max=None)
            for i in self.feature_point_card["point"].values
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
