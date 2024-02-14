# -*- coding: utf-8 -*-
''' Filters.
'''

from fenics import *
from fenics_adjoint import *
import numpy as np


def helmholtz_filter(u: Function, R=0.025) -> Function:
    ''' Apply the helmholtz filter.

    Args:
        u (dolfin_adjoint.Function): Target function.
        R (float, optional): Filter radius. Defaults to 0.025.

    Returns:
        (dolfin_adjoint.Function): Filtered function
    '''
    U = u.function_space()
    v = TrialFunction(U)
    dv = TestFunction(U)
    r = R/(2*np.sqrt(3))
    a = r**2*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
    L = inner(u, dv)*dx
    solve(a == L, u)
    return u
