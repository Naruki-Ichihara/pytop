# -*- coding: utf-8 -*-
''' Filters.
'''

from fenics import *
from fenics_adjoint import *
import numpy as np
from typing import Optional


def helmholtz_filter(u: Function,
                     R=0.025,
                     solver_parameters = Optional[any]) -> Function:
    ''' Apply the helmholtz filter to the fenics function.
    This filter directly solves the Helmholtz equation using linear solvers.
    
    Args:
        u (dolfin_adjoint.Function): Target function.
        R (float, optional): Filter radius. Defaults to 0.025.
        solver_parameters (Optional[any], optional): Solver parameters. Defaults to None.

    Returns:
        (dolfin_adjoint.Function): Filtered function
    '''
    U = u.function_space()
    v = TrialFunction(U)
    dv = TestFunction(U)
    uh = Function(U)
    r = R/(2*np.sqrt(3))
    a = r**2*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
    L = inner(u, dv)*dx
    if solver_parameters is None:
        solve(a == L, uh)
    else:
        solve(a == L, uh, solver_parameters=solver_parameters) #TODO
    u_projected = project(uh, U)
    return u_projected
