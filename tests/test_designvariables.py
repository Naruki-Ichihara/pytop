import numpy as np
import pytest
from fenics import *
from fenics_adjoint import *
from pytop.utils import fenics_function_to_np_array, np_array_to_fenics_function, create_initialized_fenics_function, set_fields_to_fenics_function
from pytop.designvector import DesignVariables
    
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
design_variables = DesignVariables()
design_variables.register(V, "test_1", [lambda x: np.sin(x[0])], (-1, 1))
design_variables.register(V, "test_2", [lambda x: np.cos(x[1])], (-1, 1))

print(design_variables.vector)
numpy_array_test_1 = np.ones(design_variables.vector.size//2)*2
numpy_array_test_2 = np.ones(design_variables.vector.size//2)*3
numpy_array = np.concatenate((numpy_array_test_1, numpy_array_test_2), axis=0)
print(numpy_array)
design_variables.vector = numpy_array

print(design_variables["test_1"])