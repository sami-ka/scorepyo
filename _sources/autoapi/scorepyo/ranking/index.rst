:py:mod:`scorepyo.ranking`
==========================

.. py:module:: scorepyo.ranking

.. autoapi-nested-parse::

   Classes for binary features rankers



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   scorepyo.ranking.Ranker
   scorepyo.ranking.LogOddsDensity
   scorepyo.ranking.DiverseLogOddsDensity
   scorepyo.ranking.CumulativeMetric
   scorepyo.ranking.BordaRank
   scorepyo.ranking.LassoPathRank
   scorepyo.ranking.LarsPathRank
   scorepyo.ranking.OMPRank
   scorepyo.ranking.FasterRiskRank




.. py:class:: Ranker

   Bases: :py:obj:`abc.ABC`

   Base class for binary features ranker

   Child classes must implement the _compute_ranking_features method.


   .. py:method:: compute_ranking_features(df: pandas.DataFrame, *args: Any, **kwargs: Any) -> pandas.Series


   .. py:method:: _compute_ranking_features(**kwargs: Any) -> pandas.Series
      :abstractmethod:



.. py:class:: LogOddsDensity(**kwargs)

   Bases: :py:obj:`Ranker`

   Binary features ranker based on logodds contribution of binary feature,
   and reduced importance of density of the binary feature (number of samples with a positive value on the binary feature)



   Child class of Ranker.

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, *args: Any, **kwargs: Any) -> pandas.Series

      This method ranks the binary features based on the product of :
       - logodds contribution of the binary feature
       - reduced importance of density

      This prevents having a too large emphasis on a binary feature with a large density and low log odd contribution
      Args:
          df (pd.DataFrame): DataFrame with logodds and density information, with the binary feature name as an index

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



.. py:class:: DiverseLogOddsDensity(rank_diversity: int = 1, **kwargs)

   Bases: :py:obj:`Ranker`

   Binary features ranker based on logodds contribution of binary feature,
   and density of the binary feature (number of samples with a positive value on the binary feature)

   In order to diversify the ranking, a small number of binary features coming from the same origin will be at the top of the ranking


   Child class of Ranker.

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, *args: Any, **kwargs: Any) -> pandas.Series

      This method ranks the binary features based on the product of :
       - logodds contribution of the binary feature
       - importance of density
       - origin of binary features

       If several binary features are at the top of the ranking, only self.rank_diversity+1 will be kept

      Args:
          df (pd.DataFrame): DataFrame with logodds, density and origin information, with the binary feature name as an index

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



.. py:class:: CumulativeMetric(metric, ranker: Ranker, **kwargs)

   Bases: :py:obj:`Ranker`

   Binary features ranker based on computing a metric on a growing number of binary features.

   This ranker initially sorts binary features depending on a specified ranker.
   Then it computes a classification metric by adding one log odd contribution of a binary feature at a time.
   The binary features are then ranked according to the incremental difference they made on the metric.

   Child class of Ranker

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, X_binarized: pandas.DataFrame, y: pandas.Series, *args: Any, **kwargs: Any) -> pandas.Series

      This ranker initially sorts binary features depending on a specified ranker.
      Then it computes a classification metric by adding one log odd contribution of a binary feature at a time.
      The binary features are then ranked according to the magnitude of the incremental difference they made on the metric.

      Args:
          df (pd.DataFrame): information dataframe, it should contain the log_odds contribution of binary feature
          X_binarized (pd.DataFrame): binary dataset
          y (pd.Series): binary target

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



.. py:class:: BordaRank(list_ranker: List[Ranker], **kwargs)

   Bases: :py:obj:`Ranker`

   Based on a list of Ranker, computes the Borda rank of each binary feature based on all rankers.

   Borda rank : https://en.wikipedia.org/wiki/Borda_count

   Child class of ranker

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, *args: Any, **kwargs: Any) -> pandas.Series

      Based on a list of Ranker, computes the Borda rank of each binary feature based on all rankers.

      Borda rank : https://en.wikipedia.org/wiki/Borda_count

      Args:
          df (pd.DataFrame): DataFrame of logodds, density, features and binary features information needed for all rankers

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



.. py:class:: LassoPathRank(**kwargs)

   Bases: :py:obj:`Ranker`

   Binary feature ranker based on lasso path.

   Lasso path stores the coefficient values of features along different values of the regularization parameters,
   of a Lasso regression.
   Based on this, the ranker selects the lasso path step where the number of binary features with non-zero coefficient
   is equal to the specified target.

   For more info on Lasso path : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html



   Child class of Ranker

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, X_binarized: pandas.DataFrame, y: pandas.Series, nb_steps: int, **kwargs) -> pandas.Series

      This functions returns the rank according to the lasso path
      Args:
          df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
          X_binarized (pd.DataFrame): Binary features
          y (pd.Series): Binary target
          nb_steps (int): Number of non zero coefficient to identify the step in the lasso path

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



.. py:class:: LarsPathRank(**kwargs)

   Bases: :py:obj:`Ranker`

   Binary feature ranker based on LARS lasso path.

   LARS lasso path is the coefficient values of features along different values of the regularization parameters,
   of a LARS Lasso regression.
   Based on this, the ranker selects the LARS lasso path step where the number of binary features with non-zero coefficient
   is equal to the specified target.

   For more info on LARS Lasso path : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path


   Child class of Ranker

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, X_binarized: pandas.DataFrame, y: pandas.Series, nb_steps: int, **kwargs) -> pandas.Series

      This function returns binary features rank based on LARS Lasso path

      Args:
          df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
          X_binarized (pd.DataFrame): Binary features
          y (pd.Series): Binary target
          nb_steps (int): Number of non zero coefficient to identify the step in the LARS lasso path

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



.. py:class:: OMPRank(**kwargs)

   Bases: :py:obj:`Ranker`

   Binary feature ranker based on Orthogonal Matching Pursuit.

      From https://scikit-learn.org/stable/modules/linear_model.html#omp:
      " Orthogonal Matching Pursuit algorithm  approximates the fit of a linear model with constraints imposed on the number of non-zero coefficients (ie. the
   pseudo-norm).

      Based on this, the ranker selects the binary features selected by the OMP algorithm where the number of non zero coefficients is equal to the specified target.

      Child class of Ranker.

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, X_binarized: pandas.DataFrame, y: pandas.Series, nb_steps: int, **kwargs) -> pandas.Series

      This function returns binary features rank based on OMP

      Args:
          df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
          X_binarized (pd.DataFrame): Binary features
          y (pd.Series): Binary target
          nb_steps (int): Number of non zero coefficient to identify the step in the LARS lasso path

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



.. py:class:: FasterRiskRank(parent_size=10, child_size=None, max_attempts=50, num_ray=20, lineSearch_early_stop_tolerance=0.001, min_point_value=-2, max_point_value=3, nb_max_features=4, **kwargs)

   Bases: :py:obj:`Ranker`

   Binary features ranker based on FasterRisk, another risk score model library.

   FasterRisk has its own algorithm to select candidate binary features.
   Based on their algorithm, we take the top features based on number of appearances in candidate models.
   Then we sort by the product of density and lod odds of binary features

   For more information of FasterRisk package and usage: https://fasterrisk.readthedocs.io/en/latest/

   Child class of Ranker

   .. py:method:: _compute_ranking_features(df: pandas.DataFrame, X_binarized: pandas.DataFrame, y: pandas.Series, nb_steps: int, **kwargs) -> pandas.Series

      This function returns binary features rank based on number of appearances in FasterRisk candidate risk score models

      Args:
          df (pd.DataFrame): Information dataframe on logodds, density and binary feature name
          X_binarized (pd.DataFrame): Binary features
          y (pd.Series): Binary target
          nb_steps (int): Number of non zero coefficient to identify the step in the LARS lasso path

      Returns:
          pd.Series: Rank for each binary feature in a decreasing order of importance



