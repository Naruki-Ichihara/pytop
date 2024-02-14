import pytop as pt
import numpy as np

pt.parameters["form_compiler"]["optimize"] = True
pt.parameters["form_compiler"]["cpp_optimize"] = True
pt.parameters['form_compiler']['quadrature_degree'] = 5

# parameters
TARGET_DENSITY = 0.3
FILTER_RADIUS = 0.1
NUMBER_OF_NODES = 200

# SIMP
def simp(rho, p=3, eps=1e-3):
    return eps + (1 - eps) * rho ** p

mesh = pt.UnitSquareMesh(NUMBER_OF_NODES, NUMBER_OF_NODES)
U = pt.FunctionSpace(mesh, "CG", 1)
uh = pt.Function(U)
u = pt.TrialFunction(U)
du = pt.TestFunction(U)
f = pt.interpolate(pt.Constant(1e-2), U)

class Left(pt.SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/10 + 1e-5
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary
bc = pt.DirichletBC(U, pt.Constant(0.0), Left())

design_variables = pt.DesignVariables()
design_variables.register(U,
                          "density",
                          [TARGET_DENSITY],
                          (0, 1),
                          recording_path="output")

class Problem(pt.ProblemStatement):
    def objective(self, design_variables):
        rho = design_variables["density"]
        rho = pt.helmholtz_filter(rho, FILTER_RADIUS)
        a = pt.inner(pt.grad(u), simp(rho)*pt.grad(du)) * pt.dx
        L = pt.inner(f, du) * pt.dx
        pt.solve(a == L, uh, bc)
        return pt.assemble(pt.inner(f, uh) * pt.dx)
    
    def constraint_volume(self, design_variables):
        rho = design_variables["density"]
        unitary = pt.project(pt.Constant(1), U)
        return pt.assemble(rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY
    
problem = Problem()
problem.objective(design_variables)

opt = pt.NloptOptimizer(design_variables, problem)
opt.run()