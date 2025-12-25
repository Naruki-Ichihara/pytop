# -*- coding: utf-8 -*-
''' Filters.
'''

from fenics import *
from fenics_adjoint import *
import numpy as np
from typing import Optional
from ufl import tanh

def helmholtz_filter(u: Function,
                     R=0.025,
                     solver_parameters : Optional[any] = None) -> Function:
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
    # Use LinearVariationalProblem and LinearVariationalSolver for better compatibility with fenics_adjoint
    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem)
    if solver_parameters is not None:
        solver.parameters.update(solver_parameters)
    solver.solve()
    u_projected = project(uh, U)
    return u_projected

def smooth_heviside_filter(u: Function, beta: float=10.0, eta: float=0.5) -> Function:
    ''' Apply the smoothed Heaviside filter to the fenics function.
    
    Args:
        u (dolfin_adjoint.Function): Target function.
        beta (float, optional): Smoothing parameter. Defaults to 10.0.
        eta (float, optional): Threshold value. Defaults to 0.5.

    Returns:
        (dolfin_adjoint.Function): Filtered function
    '''
    function_space = u.function_space()
    return project((tanh(beta*eta)+tanh(beta*(u-eta)))/(tanh(beta*eta)+tanh(beta*(1.0-eta))), function_space)