import pytop as pt
mpi_communicator = pt.MPI.comm_world

pt.parameters["form_compiler"]["optimize"] = True
pt.parameters["form_compiler"]["cpp_optimize"] = True
pt.parameters['form_compiler']['quadrature_degree'] = 5

solver_parameters = {'linear_solver': 'cg'}

# parameters
TARGET_DENSITY = 0.3
FILTER_RADIUS = 0.1
NUMBER_OF_NODES = (50, )*3

# SIMP
def simp(rho, p=3, eps=1e-3):
    return eps + (1 - eps) * rho ** p

mesh = pt.UnitCubeMesh(mpi_communicator, *NUMBER_OF_NODES)
U = pt.FunctionSpace(mesh, "CG", 1)
uh = pt.Function(U)
u = pt.TrialFunction(U)
du = pt.TestFunction(U)
f = pt.interpolate(pt.Constant(1e-2), U)

class D_bc(pt.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < 0.01 and 0.4 < x[1] < 0.6 and 0.4 < x[2] < 0.6) and on_boundary
bc = pt.DirichletBC(U, pt.Constant(0.0), D_bc())

design_variables = pt.DesignVariables()
design_variables.register(U,
                          "density",
                          [TARGET_DENSITY],
                          (0, 1),
                          lambda x: pt.helmholtz_filter(x, FILTER_RADIUS, solver_parameters),
                          recording_path="output",
                          recording_interval=1)

class Problem(pt.ProblemStatement):
    def objective(self, design_variables):
        rho = design_variables["density"]
        a = pt.inner(pt.grad(u), simp(rho)*pt.grad(du)) * pt.dx
        L = pt.inner(f, du) * pt.dx
        pt.solve(a == L, uh, bc)
        return pt.assemble(pt.inner(f, uh) * pt.dx)
    
    def constraint_volume(self, design_variables):
        rho = design_variables["density"]
        unitary = pt.project(pt.Constant(1), U)
        return pt.assemble(rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY

opt = pt.NloptOptimizer(design_variables, Problem(), "LD_MMA")
opt.set_maxeval(100)
opt.set_ftol_rel(1e-5)
opt.run("output/logging.csv")