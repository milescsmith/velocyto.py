from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from math import isclose

from numpy import arange


MKL_BUG_ERROR_MSG = """
Your current Python installation is affected by a critical bug in numpy and
MKL, and is going to return wrong results in velocount and potentially other
scientific packages.

Please try updating your `numpy` version.

For more information, see
https://github.com/velocount-team/velocount.py/issues/104
and
https://github.com/ContinuumIO/anaconda-issues/issues/10089
"""

std_check = arange(1000000).std()
expected = 288675.1345946685

if not isclose(std_check, expected):
    raise RuntimeError(MKL_BUG_ERROR_MSG)
