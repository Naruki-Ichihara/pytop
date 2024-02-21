import numpy as np
import pytest
from fenics import *
from fenics_adjoint import *
from pytop.filters import helmholtz_filter
from pytop.utils import create_initialized_fenics_function

def test_helmholtz_filter():
    mesh = UnitSquareMesh(10, 10)
    function_space = FunctionSpace(mesh, "CG", 1)
    test_function = create_initialized_fenics_function([lambda x: sin(x[0])], function_space)
    u_filtered = helmholtz_filter(test_function)
    assert isinstance(u_filtered, Function)