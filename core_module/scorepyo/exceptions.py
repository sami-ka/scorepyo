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

    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


class NegativeValueError(Exception):
    """Error raised when quantity value is not a strictly positive float"""

    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


class NonIntegerValueError(Exception):
    """Error raised when value is not an integer"""

    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)
