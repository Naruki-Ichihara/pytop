# -*- coding: utf-8 -*-
''' Filters.
'''

from fenics import *
from fenics_adjoint import *
import numpy as np


def helmholtzFilter(u: Function, U: FunctionSpace, R=0.025) -> Function:
    ''' Apply the helmholtz filter.

    Args:
        u (dolfin_adjoint.Function): Target function
        U (dolfin_adjoint.FunctionSpace): Functionspace for the helmholtz equation
        R (float, optional): Filter radius. Defaults to 0.025.

    Returns:
        (dolfin_adjoint.Function): Filtered function
    '''
    projectedFunctionOnU = project(u, U)
    v = TrialFunction(U)
    dv = TestFunction(U)
    filterdFunctionOnU = Function(U)
    r = R/(2*np.sqrt(3))
    a = r**2*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
    L = inner(projectedFunctionOnU, dv)*dx
    solve(a == L, filterdFunctionOnU,
          solver_parameters={"linear_solver": "lu"},
          form_compiler_parameters={"optimize": True})
    filterdFunctionOnOrigin = project(filterdFunctionOnU, u.function_space())
    return filterdFunctionOnOrigin
