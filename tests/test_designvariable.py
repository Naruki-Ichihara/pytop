import numpy as np
import pytest
from fenics import *
from fenics_adjoint import *
from pytop.designvariable import DesignVariables
from pytop.utils import create_initialized_fenics_function, set_fields_to_fenics_function


def test_designvariables_register():
    function_space_test_1 = FunctionSpace(UnitSquareMesh(10, 10), "CG", 1)
    function_space_test_2 = VectorFunctionSpace(UnitSquareMesh(10, 10), "CG", 1)
    design_variables = DesignVariables()

    design_variables.register(function_space_test_1,
                              "test_1",
                              [lambda x: sin(x[0])],
                              [(-1, 1)])
    assert len(design_variables) == 121
    assert isinstance(design_variables["test_1"], Function)
    assert "test_1" in design_variables
    assert "test_2" not in design_variables

    design_variables.register(function_space_test_2,
                              "test_2",
                              [lambda x: sin(x[0]), lambda x: sin(x[1])],
                              [(-1, 1), (-1, 1)])
    assert len(design_variables) == 363
    assert isinstance(design_variables["test_2"], Function)

    # Test: Registering a design variable with the same name
    with pytest.raises(ValueError):
        design_variables.register(function_space_test_1,
                                  "test_2",
                                  [lambda x: sin(x[0])],
                                  [(-1, 1)])
    
    # Test: The initial value of the design variable is not within the bounds
    with pytest.raises(ValueError):
        design_variables.register(function_space_test_1,
                                  "test_3",
                                  [lambda x: -1],
                                  [(0, 1)])

    with pytest.raises(ValueError):
        design_variables.register(function_space_test_1,
                                  "test_4",
                                  [lambda x: 1],
                                  [(-1, 0)])

def test_designvariables_set_values():
    function_space_test = FunctionSpace(UnitSquareMesh(10, 10), "CG", 1)
    design_variables = DesignVariables()

    design_variables.register(function_space_test,
                              "test_1",
                              [lambda x: sin(x[0])],
                              [(-1, 1)])
    
    assert np.array_equal(design_variables["test_1"].vector().get_local(),
                          design_variables.vector)
    
    with pytest.raises(TypeError):
        design_variables["test_2"] = create_initialized_fenics_function([0], function_space_test)

    numpy_one_array = np.ones(len(design_variables))
    design_variables.vector = numpy_one_array

    assert np.array_equal(design_variables["test_1"].vector().get_local(),
                          numpy_one_array)