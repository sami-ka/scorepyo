:py:mod:`scorepyo._utils`
=========================

.. py:module:: scorepyo._utils

.. autoapi-nested-parse::

   Fast Numba ROC AUC

   credit : https://github.com/diditforlulz273/fastauc/blob/main/fastauc/fast_auc.py



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   scorepyo._utils.fast_numba_auc
   scorepyo._utils.trapezoid_area
   scorepyo._utils.fast_numba_auc_nonw
   scorepyo._utils.fast_numba_auc_w



.. py:function:: fast_numba_auc(y_true: numpy.typing.NDArray, y_score: numpy.typing.NDArray, sample_weight: numpy.typing.NDArray = None) -> float

   a function to calculate AUC via python + numba.
   Args:
       y_true (np.array): 1D numpy array as true labels.
       y_score (np.array): 1D numpy array as probability predictions.
       sample_weight (np.array): 1D numpy array as sample weights, optional.
   Returns:
       AUC score as float


.. py:function:: trapezoid_area(x1: float, x2: float, y1: float, y2: float) -> float


.. py:function:: fast_numba_auc_nonw(y_true: numpy.typing.NDArray, y_score: numpy.typing.NDArray) -> float


.. py:function:: fast_numba_auc_w(y_true: numpy.typing.NDArray, y_score: numpy.typing.NDArray, sample_weight: numpy.typing.NDArray) -> float


