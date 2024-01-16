import numpy as np
import pytest
from fenics import *
from fenics_adjoint import *
from pytop import DesignVariables
from pytop.utils import fenics2np, np2fenics, setValuesToFunction, createInitializedFunction

class field2D(UserExpression):
    def eval(self, value, x):
        value[0] = sin(x[0])
    def value_shape(self):
        return ()

def test_utils():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    Vdouble = FunctionSpace(mesh, "CG", 2)

    constant = Constant(1.0)
    result = fenics2np(constant)

    # fenics2np - Constant
    assert np.array_equal(result, np.array([1.0]))

    # fenics2np - Function
    func = Function(V)
    func.interpolate(field2D())
    result = fenics2np(func)
    assert isinstance(result, np.ndarray)
    assert result.shape == (121,)

    # fenics2np - GenericVector and np2fenics
    vec = func.vector()
    npVector = fenics2np(vec)
    fenicsResult = np2fenics(npVector, func)
    assert np.array_equal(fenicsResult.vector().get_local(), npVector)

    # Errors
    with pytest.raises(TypeError):
        fenics2np("invalid_input")
        np2fenics("invalid_input", "invalid_input")
    with pytest.raises(ValueError):
        npArray = fenics2np(func)
        np2fenics(npArray, Function(Vdouble))

def test_createInitializedFunction():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    sin_func = createInitializedFunction([lambda x: sin(x[0])], V)
    assert isinstance(sin_func, Function)

    class field1D(UserExpression):
        def eval(self, value, x):
            value[0] = sin(x[0])
        def value_shape(self):
            return ()
        
    test_func = Function(V)
    test_func.interpolate(field1D())
    assert np.array_equal(sin_func.vector().get_local(), test_func.vector().get_local())

    # 2D case
    V2D = VectorFunctionSpace(mesh, "CG", 1)
    sin_func = createInitializedFunction([lambda x: sin(x[0]), lambda x: cos(x[1])], V2D)
    assert isinstance(sin_func, Function)

    class field2D(UserExpression):
        def eval(self, value, x):
            value[0] = sin(x[0])
            value[1] = cos(x[1])
        def value_shape(self):
            return (2,)
        
    test_func = Function(V2D)
    test_func.interpolate(field2D())
    assert np.array_equal(sin_func.vector().get_local(), test_func.vector().get_local())

    vriables = {'test1': [createInitializedFunction([-2], V),
                          createInitializedFunction([lambda x: sin(x[0])], V),
                          createInitializedFunction([+2], V)]}
    
    dv = DesignVariables(variables=vriables)
    print(dv)
    print(type(dv["test1"]))

test_createInitializedFunction()