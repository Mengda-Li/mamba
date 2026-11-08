"""Supplemental rebuttal experiments; runnable as scripts or Python modules."""
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
