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
   scorepyo.models.RiskScore
   scorepyo.models.EBMRiskScore




.. py:class:: _BaseRiskScore(nb_max_features: int = 4, min_point_value: int = -2, max_point_value: int = 3)

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
      :value: 'Description'

      Column name for binary feature in the risk score summary


   .. py:attribute:: _POINT_COL
      :value: 'Point(s)'

      Column name for points in the risk score summary


   .. py:attribute:: _FEATURE_COL
      :value: 'Feature'

      Column name for original feature in the risk score summary


   .. py:method:: fit(X: pandas.DataFrame, y: pandas.Series, *args: Any, **kwargs: Any) -> Optional[NotImplementedError]
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



.. py:class:: RiskScore(binarizer: scorepyo.binarizers.BinarizerProtocol, nb_max_features: int = 4, min_point_value: int = -2, max_point_value: int = 3, nb_additional_features: int = 4, ranker: scorepyo.ranking.Ranker = OMPRank(), calibrator: scorepyo.calibration.Calibrator = VanillaCalibrator(), enumeration_maximization_metric=fast_numba_auc)

   Bases: :py:obj:`_BaseRiskScore`

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



   Methods
   -------
   fit(self, X, y):
       function creating the feature-point card and score card attributes

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

   .. py:method:: fit(X: pandas.DataFrame, y: pandas.Series, X_calib: pandas.DataFrame = None, y_calib: pandas.Series = None, categorical_features='auto', fit_binarizer=True)

      Function that search best parameters (choice of binary features, points and probabilities) of a risk score model.


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
          X_calib (pandas.DataFrame): Dataset of features to calibrate probabilities on
          y_calib (pandas.Series): Target binary values for calibration
          categorical_features: list of categorical features for the binarizer
          fit_binarizer: boolean to indicate the binarizer should be fitted or not



   .. py:method:: predict_proba(X: pandas.DataFrame) -> numpy.ndarray

      Function that outputs probability of positive class according to risk-score model

      Args:
          X (pandas.DataFrame): dataset of features

      Returns:
          ndarray of shape (n_samples, 2): probability of negative and positive class in each column resp.



.. py:class:: EBMRiskScore(nb_max_features: int = 4, min_point_value: int = -2, max_point_value: int = 3, max_number_binaries_by_features: int = 3, keep_negative: bool = True, nb_additional_features: Optional[int] = 4, ranker: scorepyo.ranking.Ranker = OMPRank(), calibrator: scorepyo.calibration.Calibrator = VanillaCalibrator(), enumeration_maximization_metric=fast_numba_auc)

   Bases: :py:obj:`RiskScore`

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



   Methods
   -------
   fit(self, X, y):
       function creating the feature-point card and score card attributes



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


