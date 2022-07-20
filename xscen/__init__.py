"""Top-level package for xscen."""
import json
import re
import warnings
from pathlib import Path
from types import ModuleType

__author__ = """Gabriel Rondeau-Genesse"""
__email__ = "rondeau-genesse.gabriel@ouranos.ca"
__version__ = "__version__ = '0.2.2'"


# monkeypatch so that warnings.warn() doesn't mention itself
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line


# Read CVs and fill a virtual module
CV = ModuleType(
    "CV",
    (
        "Mappings of (controlled) vocabulary. This module is generated automatically "
        "from json files in xscen/CVs. Functions are essentially mappings, most of "
        "which are meant to provide translations between columns.\n\n"
        "Json files must be shallow dictionaries to be supported. If the json file "
        "contains a ``is_regex: True`` entry, then the keys are automatically "
        "translated as regex patterns and the function returns the value of the first "
        "key that matches the pattern. Otherwise the function essentially acts like a "
        "normal dictionary. The 'raw' data parsed from the json file is added in the "
        "``dict`` attribute of the function."
    ),
)


def __read_CVs(cvfile):
    with cvfile.open("r") as f:
        cv = json.load(f)
    is_regex = cv.pop("is_regex", False)
    doc = """Controlled vocabulary mapping from {name}.

    The raw dictionary can be accessed by the dict attribute of this function.

    Parameters
    ----------
    key: str
      The value to translate.{regex}
    default : 'pass', 'error' or Any
      If the key is not found in the mapping, default controls the behaviour.

      - "error", a KeyError is raised (default).
      - "pass", the key is returned.
      - another value, that value is returned.
"""

    def cvfunc(key, default="error"):
        if is_regex:
            for cin, cout in cv.items():
                try:
                    if re.fullmatch(cin, key):
                        return cout
                except TypeError:
                    pass
        else:
            if key in cv:
                return cv[key]
        if isinstance(default, str):
            if default == "pass":
                return key
            if default == "error":
                raise KeyError(key)
        return default

    cvfunc.__name__ = cvfile.stem
    cvfunc.__doc__ = doc.format(
        name=cvfile.stem.replace("_", " "),
        regex=" The key will be matched using regex" if is_regex else "",
    )
    cvfunc.__dict__["dict"] = cv
    cvfunc.__module__ = "xscen.CV"
    return cvfunc


for cvfile in (Path(__file__).parent / "CVs").glob("*.json"):
    try:
        CV.__dict__[cvfile.stem] = __read_CVs(cvfile)
    except Exception as err:
        raise ValueError(f"While reading {cvfile} got {err}")
