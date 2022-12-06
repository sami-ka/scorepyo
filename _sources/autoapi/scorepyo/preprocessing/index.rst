:py:mod:`scorepyo.preprocessing`
================================

.. py:module:: scorepyo.preprocessing

.. autoapi-nested-parse::

   Class for the EBM-based automatic binarizer



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   scorepyo.preprocessing.AutoBinarizer




.. py:class:: AutoBinarizer(max_number_binaries_by_features: int = 2, keep_negative: bool = True)

   Class for automatic feature binarizer.

   This class uses Explainable Boosting Machine (EBM) that are part of the General Additive Model (GAM) family.
   EBM will compute for each feature a tree. The final prediction will be made by summing up the tree value for each feature.
   Since it's single-feature tree, it actually is a piecewise constant scalar function.
   For each interval where the single-feature tree yields a constant contribution for the prediction, this class computes a binary feature.

   Attributes
   ----------
   max_number_binaries_by_features : int
       maximum number of binary features to compute by feature
   keep_negative : bool
       indicator to keep features that decrease predicted probability
   _ebm: ExplainableBoostingClassifier
       EBM classifier that is fitted and used to binarize feature
   _one_hot_encoder: OneHotEncoder
       Scikit-learn one-hot encoder for categorical features
   _categorical_features: list(str)
       list of categorical features
   _continuous_features: list(str)
       list of continuous features
   _to_exclude_features: list(str)
       list of features to exclude from binarizing


   Methods
   -------
   fit(X, y, categorical_features="auto", to_exclude_features=None))
       fits the EBM model on X,y and the one-hot encoder on the categorical features

   transform(X)
       transforms features in X into binarized features based on previously fitted EBM and one-hot encoder

   .. py:method:: fit(X: pandas.DataFrame, y: pandas.Series, categorical_features: Union[list[str], str] = 'auto', to_exclude_features: Optional[list[str]] = None)

      Fit function of binarizer

      This functions fits the EBM on X,y and the one-hot encoder on X.
      It stores the categorical, continuous and to exclude columns.

      Args:
          X (pandas.Dataframe): Dataframe of features to use to fit the binarizer
          y (pandas.Series): target values
          categorical_features (list(str), optional): list of categorical features to one-hot encode.
          Defaults to "auto" to automatically detect categorical features.
          to_exclude_features (list(str), optional): List of features to leave as is.
          Defaults to None.


   .. py:method:: transform(X: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]

      Transform function of binarizer

      This function uses the previously fitted EBM to extract binary features from continuous features.
      For each continuous feature, it looks at each constructed interval to create a binary feature based on feature value belonging to this interval or not.
      For categorical features, it uses the one-hot encoder previously fitted.
      For features to exclude from the binarizer, it copies the values in the new dataset.

      Args:
          X (pandas.DataFrame): Dataframe of features to transform

      Returns:
          pandas.DataFrame: Binarized features
          pandas.DataFrame: DataFrame of information of binary feature and corresponding feature



