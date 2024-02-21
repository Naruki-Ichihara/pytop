import numpy as np
import pytest
from fenics import *
from fenics_adjoint import *
from pytop.utils import fenics_function_to_np_array, np_array_to_fenics_function, create_initialized_fenics_function, set_fields_to_fenics_function


class Field2D(UserExpression):
    def eval(self, value, x):
        value[0] = sin(x[0])

    def value_shape(self):
        return ()

def test_conversion():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    Vdouble = FunctionSpace(mesh, "CG", 2)

    constant = Constant(1.0)
    result = fenics_function_to_np_array(constant)

    # fenics2np - Constant
    assert np.array_equal(result, np.array([1.0]))

    # fenics2np - Function
    func = Function(V)
    func.interpolate(Field2D())
    result = fenics_function_to_np_array(func)
    assert isinstance(result, np.ndarray)
    assert result.shape == (121,)

    # fenics2np - GenericVector and np2fenics
    vec = func.vector()
    np_vector = fenics_function_to_np_array(vec)
    fenics_function = np_array_to_fenics_function(np_vector, func)
    assert np.array_equal(fenics_function.vector().get_local(), np_vector)
    assert np.array_equal(fenics_function.vector().get_local(), vec.get_local())

    # Errors
    with pytest.raises(TypeError):
        fenics_function_to_np_array("invalid_input")
        np_array_to_fenics_function("invalid_input", "invalid_input")
    with pytest.raises(ValueError):
        np_array = fenics_function_to_np_array(func)
        np_array_to_fenics_function(np_array, Function(Vdouble))

def test_set_fields_to_fenics_function():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)

    set_fields_to_fenics_function([lambda x: sin(x[0])], v)

    class field1D(UserExpression):
        def eval(self, value, x):
            value[0] = sin(x[0])

        def value_shape(self):
            return ()

    test_func = Function(V)
    test_func.interpolate(field1D())
    assert np.array_equal(v.vector().get_local(),
                          test_func.vector().get_local())

    # 2D case
    V2D = VectorFunctionSpace(mesh, "CG", 1)
    v2D = Function(V2D)
    set_fields_to_fenics_function(
        [lambda x: sin(x[0]), lambda x: cos(x[1])], v2D)

    class field2D(UserExpression):
        def eval(self, value, x):
            value[0] = sin(x[0])
            value[1] = cos(x[1])

        def value_shape(self):
            return (2,)

    test_func = Function(V2D)
    test_func.interpolate(field2D())
    assert np.array_equal(v2D.vector().get_local(),
                          test_func.vector().get_local())


def test_create_initialized_fenics_function():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    sin_func = create_initialized_fenics_function([lambda x: sin(x[0])], V)
    assert isinstance(sin_func, Function)

    class field1D(UserExpression):
        def eval(self, value, x):
            value[0] = sin(x[0])

        def value_shape(self):
            return ()

    test_func = Function(V)
    test_func.interpolate(field1D())
    assert np.array_equal(sin_func.vector().get_local(),
                          test_func.vector().get_local())

    # 2D case
    V2D = VectorFunctionSpace(mesh, "CG", 1)
    sin_func = create_initialized_fenics_function(
        [lambda x: sin(x[0]), lambda x: cos(x[1])], V2D)
    assert isinstance(sin_func, Function)

    class field2D(UserExpression):
        def eval(self, value, x):
            value[0] = sin(x[0])
            value[1] = cos(x[1])

        def value_shape(self):
            return (2,)

    test_func = Function(V2D)
    test_func.interpolate(field2D())
    assert np.array_equal(sin_func.vector().get_local(),
                          test_func.vector().get_local())