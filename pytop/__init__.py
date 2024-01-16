"""
.. include:: ../README.md
"""

__version__ = "0.0.0.alpha"

from fenics import *
from fenics_adjoint import *
from .utils import fenics2np, np2fenics, setValuesToFunction, createInitializedFunction
from .statement import ProblemStatement
from .designvector import DesignVariables