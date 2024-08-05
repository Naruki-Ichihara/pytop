from fenics import *
from fenics_adjoint import *
import numpy as np
from typing import Callable, Iterable, Optional
from dataclasses import dataclass
from ufl import tanh
import ufl

def penalized_weight(rho, p=3, eps=1e-3):
    '''Penalized weight function.
    
    Args: (float, float)
        rho: density.
        p: penalization parameter.
        eps: penalization parameter.
        
    Returns: (float)
        penalized weight.
    '''
    return eps + (1 - eps) * rho ** p

def sgn(x, k=10):
    return tanh(k*x)

def isoparametric_2D(z: Function, e: Function, u: Function, v: Function) -> Function:
    ''' Apply 2D isoparametric projection onto orientation vector.

    Args:
        z (dolfin_adjoint.Function): 0-component of the orientation vector (on natural setting).
        e (dolfin_adjoint.Function): 1-component of the orientation vector (on natural setting)
        u (dolfin_adjoint.Function): 0-component of the orientation vector (on real setting).
        v (dolfin_adjoint.Function): 1-component of the orientation vector (on real setting).

    Returns:
        dolfin_adjoint.Vector: Orientation vector with unit circle boundary condition on real setting.
    '''
    N1 = -(1-z)*(1-e)*(1+z+e)/4
    N2 =  (1-z**2)*(1-e)/2
    N3 = -(1+z)*(1-e)*(1-z+e)/4
    N4 =  (1+z)*(1-e**2)/2
    N5 = -(1+z)*(1+e)*(1-z-e)/4
    N6 =  (1-z**2)*(1+e)/2
    N7 = -(1-z)*(1+e)*(1+z-e)/4
    N8 =  (1-z)*(1-e**2)/2
    N = as_vector([N1, N2, N3, N4, N5, N6, N7, N8])
    Nx = inner(u, N)
    Ny = inner(v, N)
    return as_vector((Nx, Ny))

def isoparametric_2D_box_to_circle(z: Function, e: Function) -> Function:
    ''' Apply 2D isoparametric projection onto orientation vector.

    Args:
        z (dolfin_adjoint.Function): 0-component of the orientation vector (on natural setting).
        e (dolfin_adjoint.Function): 1-component of the orientation vector (on natural setting)

    Returns:
        dolfin_adjoint.Vector: Orientation vector with unit circle boundary condition on real setting.
    '''    
    u = as_vector([-1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0, -1/np.sqrt(2), -1])
    v = as_vector([-1/np.sqrt(2), -1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0])
    return isoparametric_2D(z, e, u, v)

def isoparametric_2D_box_to_triangle(z: Function, e: Function, tolerance=1e-6) -> Function:
    ''' Apply 2D isoparametric projection onto orientation vector.

    Args:
        z (dolfin_adjoint.Function): 0-component of the orientation vector (on natural setting).
        e (dolfin_adjoint.Function): 1-component of the orientation vector (on natural setting).
        tolerance (float): tolerance value. This is used to avoid square root of negative values.

    Returns:
        dolfin_adjoint.Vector: Orientation vector with unit circle boundary condition on real setting.
    '''    
    zero_like = tolerance
    u = as_vector([zero_like, 0.5, 1, 0.75, 0.5, 0.25, zero_like, zero_like])
    v = as_vector([zero_like, zero_like, zero_like, 0.25, 0.5, 0.75, 1, 0.5])
    return isoparametric_2D(z, e, u, v)

class Custom_nonlinear_problem(NonlinearProblem):
    """Custom nonlinear problem class for FEniCS.
    """
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

def inner_e(x, y, restrict_to_one_side=False, quadrature_degree=1):
    r"""The inner product of the tangential component of a vector field on all
    of the facets of the mesh (Measure objects dS and ds).
    By default, restrict_to_one_side is False. In this case, the function will
    return an integral that is restricted to both sides ('+') and ('-') of a
    shared facet between elements. You should use this in the case that you
    want to use the 'projected' version of DuranLibermanSpace.
    If restrict_to_one_side is True, then this will return an integral that is
    restricted ('+') to one side of a shared facet between elements. You should
    use this in the case that you want to use the `multipliers` version of
    DuranLibermanSpace.
    Args:
        x: DOLFIN or UFL Function of rank (2,) (vector).
        y: DOLFIN or UFL Function of rank (2,) (vector).
        restrict_to_one_side (Optional[bool]: Default is False.
        quadrature_degree (Optional[int]): Default is 1.
    Returns:
        UFL Form.
    """
    dSp = Measure('dS', metadata={'quadrature_degree': quadrature_degree})
    dsp = Measure('ds', metadata={'quadrature_degree': quadrature_degree})
    n = ufl.geometry.FacetNormal(x.ufl_domain())
    t = as_vector((-n[1], n[0]))
    a = (inner(x, t)*inner(y, t))('+')*dSp + \
        (inner(x, t)*inner(y, t))*dsp
    if not restrict_to_one_side:
        a += (inner(x, t)*inner(y, t))('-')*dSp
    return a