""" Definition of the different classes for the Error management """
from __future__ import annotations

import pandera as pa
import pandera.extensions as extensions
from pandas.api.types import is_numeric_dtype


@extensions.register_check_method()  # decorator needed by pandera
def is_numeric(pandas_obj):
    """function that returns true if series dtype is numeric

    The decorator is meant to be

    Args:
        pandas_obj (Pandas.Series): series with the datatype check

    Returns:
        bool: True if data type is numeric, False if not
    """
    return is_numeric_dtype(pandas_obj)


def NumericCheck():
    """Checks that a pandas Series has a numeric data type

    THis checks is to ensure that the rolling aggregation of pandas will run smoothly afterward

    Returns:
        pandera.Check: Pandera check
    """
    return pa.Check.is_numeric(
        title="Numeric type check",
        description="Check that the column is a numeric type to ensure pandas rolling aggregation will work",
    )


# General errors


class MissingColumnError(Exception):
    """Error raised when column is missing in a pd.DataFrame"""

    pass


class NegativeValueError(Exception):
    """Error raised when quantity value is not a strictly positive float"""

    pass


class NonIntegerValueError(Exception):
    """Error raised when value is not an integer"""

    pass


class MinPointOverMaxPointError(Exception):
    """Error raised when min point value of ScoreCard model
    is over max point value"""

    pass


class NonBooleanValueError(Exception):
    """Error raised when value is not boolean"""

    pass


class NonProbabilityValues(Exception):
    """Error raised when a probability value returned by a calibrator is not valid (between 0 and 1)"""

    pass


class NonIncreasingProbabilities(Exception):
    """Error raised when list of probabilities returned by a calibrator is not increasing"""

    pass
