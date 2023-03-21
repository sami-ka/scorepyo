:py:mod:`scorepyo.calibration`
==============================

.. py:module:: scorepyo.calibration

.. autoapi-nested-parse::

   Class for calibrators



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   scorepyo.calibration.Calibrator
   scorepyo.calibration.VanillaCalibrator
   scorepyo.calibration.BootstrappedCalibrator




.. py:class:: Calibrator(**kwargs)

   Bases: :py:obj:`abc.ABC`

   Base Class for calibrator.

   RiskScore models will assign a score to each sample. Since the scores are a simple sum of few integers,
   we can enumerate all the possible values for scores. At each score value, there are negative and positive samples.
   Perfectly calibrated probabilities on a dataset would be proportion of positives at each score.
   However, there is a need for preserving the increasing trend between score and probability.
   For that, we optimize the logloss which naturally leads to calibrated probabilities, under ordering constraints
   The optimization is based on the counting of positive and negative samples at each possible score value.
   Args:
       ABC (_type_): _description_

   .. py:method:: calibrate(df_score: pandas.DataFrame, min_sum_point: int, max_sum_point: int, **kwargs) -> pandas.DataFrame

      Funtion that takes a Dataframe of scores and binary target, and compute the associated probabilities for each sum of points

      It will use the _calibrate function defined in the child classes to compute these probabilities.

      Args:
          df_score (pd.DataFrame): DataFrame of scores and binary target
          min_sum_point (int): minimum possible sum of points
          max_sum_point (int): maximum possible sum of points

      Returns:
          pd.DataFrame: DataFrame containing the probability assigned to each possible score( or sum of points)


   .. py:method:: _calibrate(*args: Any, **kwargs: Any) -> List[float]
      :abstractmethod:



.. py:class:: VanillaCalibrator(**kwargs)

   Bases: :py:obj:`Calibrator`

   Vanilla calibrator that simply optimizes the logloss under ordering constraints.

   Given a list of increasing scores, and an associated count of negative and positive samples,
   this calibrator computes the probabilities by optimizing the logloss on the whole dataset,
   and respecting the ordering of probabilities according to scores.

   This calibrator should be favored when calibrating on a large dataset.


   .. py:method:: _calibrate(df_cvx: pandas.DataFrame, **_kwargs) -> List[float]



.. py:class:: BootstrappedCalibrator(nb_experiments: int = 20, method: str = 'average', **_kwargs)

   Bases: :py:obj:`Calibrator`

   Bootstrapped calibrator that optimizes the logloss under ordering constraints on different bootstrapped sets.

   Given an original dataset, this calibrator bootstraps several times other dataset and finds probability that optimize the logloss on all datasets,
   still respecting the probability ordering by score.
   This BootstrappedCalibrator class has two modes, it can either:
   - optimize the average logloss across all bootstrapped datasets
   - optimize the worse logloss among dataset

   The latter will lead to worse logloss on the training dataset, but more robust logloss on the test set if it's similar to the distribution on the training dataset.
   The BootstrappedCalibrator should be favored when calibrating on a small dataset.


   .. py:method:: _calibrate(df_cvx: pandas.DataFrame, **_kwargs) -> List[float]



