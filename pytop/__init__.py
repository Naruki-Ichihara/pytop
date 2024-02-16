"""
.. include:: ../README.md
"""

__version__ = "0.0.0.alpha"

from fenics import *
from fenics_adjoint import *
from .utils import fenics_function_to_np_array, np_array_to_fenics_function, create_initialized_fenics_function, set_fields_to_fenics_function
from .statement import ProblemStatement
from .designvariable import DesignVariables
from .filters import helmholtz_filter
from .nlopt_optimizer import NloptOptimizer
