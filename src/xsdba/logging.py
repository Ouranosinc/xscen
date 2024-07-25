"""
Logging utilities
================================
"""

from __future__ import annotations

import logging as _logging
import warnings

logger = _logging.getLogger("xsdba")


# XC put here to avoid circular import
class ValidationError(ValueError):
    """Error raised when input data to an indicator fails the validation tests."""

    @property
    def msg(self):  # noqa
        return self.args[0]


# XC put here to avoid circular import
def raise_warn_or_log(
    err: Exception,
    mode: str,
    msg: str | None = None,
    err_type: type = ValueError,
    stacklevel: int = 1,
):
    """Raise, warn or log an error according.

    Parameters
    ----------
    err : Exception
        An error.
    mode : {'ignore', 'log', 'warn', 'raise'}
        What to do with the error.
    msg : str, optional
        The string used when logging or warning.
        Defaults to the `msg` attr of the error (if present) or to "Failed with <err>".
    err_type : type
        The type of error/exception to raise.
    stacklevel : int
        Stacklevel when warning. Relative to the call of this function (1 is added).
    """
    message = msg or getattr(err, "msg", f"Failed with {err!r}.")
    if mode == "ignore":
        pass
    elif mode == "log":
        logger.info(message)
    elif mode == "warn":
        warnings.warn(message, stacklevel=stacklevel + 1)
    else:  # mode == "raise"
        raise err from err_type(message)
