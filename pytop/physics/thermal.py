from fenics import *
from fenics_adjoint import *
import numpy as np
from typing import Callable, Iterable, Optional
from dataclasses import dataclass

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