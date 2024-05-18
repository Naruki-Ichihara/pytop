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

def linear_2D_elasticity_bilinear_form(trial_function: TrialFunction, test_function: TestFunction, E: float, nu: float, weight: Callable=None) -> Form:
    '''Bilinear form of Linear 2D elasticity.

    Args: (TrialFunction, TestFunction, float, float, Callable)
        trial_function: trial function.
        test_function: test function.
        E: Young's modulus.
        nu: Poisson's ratio.
        weight: weight function.

    Returns: (Form)
        bilinear form.
    '''
    if weight is None:
        weight = Constant(1)
    a = inner(weight * elastic_2d_plane_stress(trial_function, E, nu), strain(test_function)) * dx
    return a


def linear_2D_poission_bilinear_form(trial_function: TrialFunction, test_function: TestFunction, weight: Callable=None) -> Form:
    '''Bilinear form of Linear 2D elasticity.

    Args: (TrialFunction, TestFunction, Callable)
        trial_function: trial function.
        test_function: test function.
        weight: weight function.

    Returns: (Form)
        bilinear form.
    '''
    if weight is None:
        weight = Constant(1)
    a = inner(weight * grad(trial_function), grad(test_function)) * dx
    return a


def strain(u):
    '''Compute the strain tensor.
    
    Args: (Function)
        u: displacement field.
        
    Returns: (Function)
        strain tensor.    
    '''
    return sym(grad(u))

def elastic_2d_plane_stress(u, E, nu):
    '''Compute the elastic stress tensor.
    
    Args: (Function, float, float)
        u: displacement field.
        E: Young's modulus.
        nu: Poisson's ratio.
        
    Returns: (Function)
        stress tensor.    
    '''
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * strain(u) + lmbda * tr(strain(u)) * Identity(u.geometric_dimension())