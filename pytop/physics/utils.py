from fenics import *
from fenics_adjoint import *
import numpy as np
from typing import Callable, Iterable, Optional
from dataclasses import dataclass

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

def sign(x, a=1):
    return ((1/(1+exp(-a*x)))-1/2)*2

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

def isoparametric_2D_box_to_triangle(z: Function, e: Function) -> Function:
    ''' Apply 2D isoparametric projection onto orientation vector.

    Args:
        z (dolfin_adjoint.Function): 0-component of the orientation vector (on natural setting).
        e (dolfin_adjoint.Function): 1-component of the orientation vector (on natural setting)

    Returns:
        dolfin_adjoint.Vector: Orientation vector with unit circle boundary condition on real setting.
    '''    
    u = as_vector([0, 0.5, 1, 0.75, 0.5, 0.25, 0, 0])
    v = as_vector([0, 0, 0, 0.25, 0.5, 0.75, 1, 0.5])
    return isoparametric_2D(z, e, u, v) 