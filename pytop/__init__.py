"""
.. include:: ../README.md
"""

__version__ = "0.0.1.alpha"

from fenics import *
from fenics_adjoint import *
#set_log_active(False)

from .utils import import_external_mesh, fenics_function_to_np_array, np_array_to_fenics_function, create_initialized_fenics_function, set_fields_to_fenics_function, MPI_Communicator, make_noiman_boundary_domains
from .statement import ProblemStatement
from .designvariable import DesignVariables
from .filters import helmholtz_filter, smooth_heviside_filter
from .nlopt_optimizer import NloptOptimizer