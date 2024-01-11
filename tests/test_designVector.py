import numpy as np
import pytest
import pytop as pt
from pytop.utils import fenics2np, np2fenics
from pytop.statement import ProblemStatement
from pytop.designvector import DesignVectorVariables

mesh10 = pt.UnitSquareMesh(1, 1)
mesh20 = pt.UnitSquareMesh(2, 2)
U = pt.FunctionSpace(mesh10, "CG", 1)
V = pt.FunctionSpace(mesh20, "CG", 1)
u = pt.Function(U)
v = pt.Function(V)
high_u = pt.Function(U)
low_u = pt.Function(U)
high_v = pt.Function(V)
low_v = pt.Function(V)

class uniformField(pt.UserExpression):
    def eval(self, value, x):
        value[0] = 0
    def value_shape(self):
        return ()
class LowUniformField(pt.UserExpression):
    def eval(self, value, x):
        value[0] = -1
    def value_shape(self):
        return ()
class HighUniformField(pt.UserExpression):
    def eval(self, value, x):
        value[0] = 1
    def value_shape(self):
        return ()

u.interpolate(uniformField())
v.interpolate(uniformField())
high_u.interpolate(HighUniformField())
low_u.interpolate(LowUniformField())
high_v.interpolate(HighUniformField())
low_v.interpolate(LowUniformField())

valiables = {'testVariable1': (low_u, u, high_u),
             'testVariable2': (low_v, v, high_v)}

dv = DesignVectorVariables(valiables)
print(dv)
print(dv.npFullsizeVector)

x = np.linspace(0, 1, dv.totalDesignVariableCount)
y = np.sin(x)*0.9
dv.npFullsizeVector = y
print(dv.npFullsizeVector)
print(dv.fenicsValues["testVariable1"].vector()[:])
dv.fieldCount