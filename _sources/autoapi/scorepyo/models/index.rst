:py:mod:`scorepyo.models`
=========================

.. py:module:: scorepyo.models

.. autoapi-nested-parse::

   Classes to create and fit risk-score type model.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   scorepyo.models._BaseRiskScore
   scorepyo.models.OptunaRiskScore




.. py:class:: _BaseRiskScore(nb_max_features: int = 4, min_point_value: int = -2, max_point_value: int = 3, df_info: Optional[pandas.DataFrame] = None)

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

   .. py:attribute:: _DESCRIPTION_COL
      :annotation: = Description

      Column name for binary feature in the risk score summary


   .. py:attribute:: _POINT_COL
      :annotation: = Point(s)

      Column name for points in the risk score summary


   .. py:attribute:: _FEATURE_COL
      :annotation: = Feature

      Column name for original feature in the risk score summary


   .. py:method:: _predict_proba_score(score: int, intercept: float) -> numpy.ndarray
      :staticmethod:

      Function that computes the logistic function value at score+intercept value

      Args:
          score (np.array(int)): sum of points coming from binary features
          intercept (float): intercept of score card. log-odds of having 0 point

      Returns:
          np.array(int): associated probability


   .. py:method:: fit(X: pandas.DataFrame, y: pandas.Series) -> Optional[NotImplementedError]
      :abstractmethod:

      Functions that creates the feature-point card and score card

      Must be defined for each child class

      Args:
          X (pandas.DataFrame): binary feature dataset
          y (pandas.Series): binary target

      Raises:
          NotImplementedError


   .. py:method:: predict(X: pandas.DataFrame, proba_threshold: float = 0.5) -> numpy.ndarray

      Predicts binary class based on probabillity threshold

      Afer computing the probability for each sample,

      Args:
          X (pd.DataFrame): _description_
          proba_threshold (float, optional): probability threshold for binary classification. Defaults to 0.5.

      Returns:
          nbarray of shape (n_samples,): predicted class based on predicted probability and threshold


   .. py:method:: predict_proba(X: pandas.DataFrame) -> numpy.ndarray

      Function that outputs probability of positive class according to risk-score model

      Args:
          X (pandas.DataFrame): dataset of features

      Returns:
          ndarray of shape (n_samples, 2): probability of negative and positive class in each column resp.


   .. py:method:: summary() -> None



.. py:class:: OptunaRiskScore(nb_max_features: int = 4, min_point_value: int = -2, max_point_value: int = 3, df_info: Optional[pandas.DataFrame] = None, optuna_optimize_params: Optional[dict] = None)

   Bases: :py:obj:`_BaseRiskScore`

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

   .. py:method:: score_logloss_objective(trial, X: pandas.DataFrame, y: pandas.Series) -> float

      Logloss objective function for Risk score exploration parameters sampled with optuna.

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


   .. py:method:: fit(X: pandas.DataFrame, y: pandas.Series) -> None

      Function that search best parameters (choice of binary features, points and intercept) of a risk score model with Optuna

      This functions calls Optuna to find the best parameters of a risk score model and then construct the feature-point card and score card attributes.

      Args:
          X (pandas.DataFrame): Dataset of features to fit the risk score model on
          y (pandas.Series): Target binary values



