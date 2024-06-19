<h1 align="center"><b>pytop</b></h1>

Try out in codespace:

<a href='https://codespaces.new/Naruki-Ichihara/pytop'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

## Significance

*pytop* is a library for the general-purpose optimization in finite element space, including topology optimization. We provides straightforward coding for complex optimization problems.
This software uses the [FEniCS](https://fenicsproject.org/) as finite element solver and [NLopt](https://github.com/stevengj/nlopt) as optimization solver.

> This software is the indipendent module of fenics and nlopt project.

> This software is based on Lagacy FEniCS (FEniCS2019.1.0). The new version, FEniCSx, is not supported.

> Only cpu based computation is supported. Now I consider developing the gpu based framework, please refer [gpytop](https://github.com/Naruki-Ichihara/gpytop).

## Documentation 🚧🚧Under construction🚧🚧
Documentation with many physics is available here:
<a href="https://naruki-ichihara.github.io/pytop_docs/"><strong>Documentation</strong></a>

## Introduction

Topology optimization is a common method for designing physic-objective-oriented structures. *pytop* enables straightforward pythonic coding for high performance
topology optimization. This software works with any general objective, physics, and (inequality) constraints, with automatic derivative.

## Hnads-on

We provide a container for this repository. The container includes python 3.11, FEniCS bundles, and NLOpt with python interface.
The container is avaiable in [dockerhub](https://hub.docker.com/repository/docker/ichiharanaruki/pytop/general).
To try out this repository, constract the codespace with following budge:

<a href='https://codespaces.new/Naruki-Ichihara/pytop'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

## Usage
### Import pytop
First you need to import *pytop* by standard style:
```python
import pytop as pt
```
In the header of pytop, all methods and class will be imprted if FEniCS was collectly installed. So, all method in FEniCS can be called directly like as:
```python
...
U = pt.FunctionSoace(mesh, "CG", 1)
u = pt.Function(U)
...
```
Please refer the [FEniCS document](https://fenicsproject.org/) to use fundamental functions in FEniCS.
### Mesh
Second, you need to construct (or import) mesh. This software provides easy method to import external mesh, ```import_external_mesh```. This method use ```meshio``` in backgroud, so any mesh type that supported by [meshio](https://github.com/nschloe/meshio) is supported.
```python
mesh = pt.impoprt_external_mesh(path_to_the_meshfile)
```
Of course you can construct mesh by built in mesh generator in FEniCS like as:
```python
NUMBER_OF_NODES = (100, 100)
POSITION = (100, 100)
mesh = pt.RectangleMesh(pt.Point(0, 0), pt.Point(*POSITION), *NUMBER_OF_NODES)
```

### Function space and Function
This is standard way in the FEniCS. You first need to define `FunctionSpace` and then `Function` (If need, `TrialFunction` and `TestFunction`).
```python
U = pt.VectorFunctionSpace(mesh, 'CG', 1) #1st order continuous Galerkin vector space
uh = pf.Function(U, name='displacement)
u = pt.TrialFunction(U)
du = pt.TestFunction(U)
```

### Boundary Condition
This is also standard procedure in the FEniCS. To define sub-domain of mesh, `pt.SubDomain` class is useful.
```python
class Left(pt.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 1e-6 and on_boundary

bc = pt.DirichletBC(U, pt.Constant((0, 0)), Left())
```
This restricts the left side boundary (x[0] < 1e-6) zero displacement.
To apply Noiman boundary condition, you can use useful method `make_noiman_boundary_domains`.
```python
class Loading(pt.SubDomain):
    def inside(self, x, on_boundary):
        reurn x[0] > 200-1e-6 and 45.0 < x[1] < 55.0 and on_boundary
    
ds = pt.make_noiman_boundary_domains(mesh, [Loading()], True)
```
This function returns `Measure` class, `ds`.

### Design variable
You need to define design variable before optimization. `DesignVariables` class can store multiple design variables with their function space, name, initial value, range, pre/post process, recording path, and recording parameter.
```python
X = pt.FunctionSpace(mesh, 'CG', 1)
design_variables = pt.DesignVariables()
design_variables.register(X,
                          "parameter_name",
                          [initial value],
                          [range],
                          lambda x: some pre process,
                          lambda x: some post process,
                          "recording_path",
                          recording_interval)
```
### Problem definition
The problem statement can be construct with `ProblemStatement`.
```python
# pytop.physics module includes useful utilities to define physics.
from pytop.physics import elasticity as el

class Elasticity_Problem(pt.ProblemStatement):
  # objective is abstruct method that must be defined.
　# In this, you need to define physics problem and return evaluation value.
    def objective(self, design_variables): 
        self.rho = design_variables["density"] # You need to access design variables through design_variables.
        a = el.linear_2D_elasticity_bilinear_form(u, du, E, nu, penalized_weight(self.rho, eps=1e-4)) # Bilinear form for 2D elastic problem
        L = pt.inner(f, du) * ds(1)  # ds(1) means first subdomain that defined by make_noiman_boundary_domains method.
        pt.solve(a == L, uh, bc)
        return pt.assemble(pt.inner(f, uh) * ds(1))
    # methods that start with "constraint_" is inequality constraint (x<=0).
    def constraint_volume(self, design_variables):
        unitary = pt.project(pt.Constant(1), X)
        return pt.assemble(self.rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY
```

### Optimizer
`NloptOptimizer` will construct NLopt `opt` class with `DesignVariables` and `ProblemStatement`. 
```python
opt = pt.NloptOptimizer(design_variables, Problem(), "LD_MMA")
# LD_MMA means Method of moving asymptotes.
```
Then you set parameter of optimizer by nlopt optimizer class methods.
Please refer avaiable method in [NLopt](https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/).
For example, here we set to maximum iteration number is 300, and relative tolerance is 1e-4:
```python
opt.set_maxeval(300)
opt.set_ftol_rel(1e-4)
```
Then optimization will start by calling `run` method.
```python
opt.run(path_for_data_logging)
```
## Tips
### Recording functions
### Filters for topology optimization
### Built-in physics
### Dehomogenization
