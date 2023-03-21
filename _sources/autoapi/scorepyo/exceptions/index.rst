:py:mod:`scorepyo.exceptions`
=============================

.. py:module:: scorepyo.exceptions

.. autoapi-nested-parse::

   Definition of the different classes for the Error management 



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   scorepyo.exceptions.is_numeric
   scorepyo.exceptions.NumericCheck



.. py:function:: is_numeric(pandas_obj)

   function that returns true if series dtype is numeric

   The decorator is meant to be

   Args:
       pandas_obj (Pandas.Series): series with the datatype check

   Returns:
       bool: True if data type is numeric, False if not


.. py:function:: NumericCheck()

   Checks that a pandas Series has a numeric data type

   THis checks is to ensure that the rolling aggregation of pandas will run smoothly afterward

   Returns:
       pandera.Check: Pandera check


.. py:exception:: MissingColumnError

   Bases: :py:obj:`Exception`

   Error raised when column is missing in a pd.DataFrame


.. py:exception:: NegativeValueError

   Bases: :py:obj:`Exception`

   Error raised when quantity value is not a strictly positive float


.. py:exception:: NonIntegerValueError

   Bases: :py:obj:`Exception`

   Error raised when value is not an integer


.. py:exception:: MinPointOverMaxPointError

   Bases: :py:obj:`Exception`

   Error raised when min point value of ScoreCard model
   is over max point value


.. py:exception:: NonBooleanValueError

   Bases: :py:obj:`Exception`

   Error raised when value is not boolean


.. py:exception:: NonProbabilityValues

   Bases: :py:obj:`Exception`

   Error raised when a probability value returned by a calibrator is not valid (between 0 and 1)


.. py:exception:: NonIncreasingProbabilities

   Bases: :py:obj:`Exception`

   Error raised when list of probabilities returned by a calibrator is not increasing


