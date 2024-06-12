import pytop as pt

# parameters
TARGET_DENSITY = 0.3
FILTER_RADIUS = 2.0
NUMBER_OF_NODES = 100
output_path = "output/poisson_problem"

# SIMP
def simp(rho, p=3, eps=1e-3):
    return eps + (1 - eps) * rho ** p

#mesh = pt.UnitSquareMesh(pt.MPI_Communicator.comm_world, NUMBER_OF_NODES, NUMBER_OF_NODES)
mesh = pt.import_external_mesh("tests/test_meshs/unstructured_tiangle_xy2D_rectangle_test_mesh.inp", pt.MPI_Communicator)
U = pt.FunctionSpace(mesh, "CG", 1)
uh = pt.Function(U)

u = pt.TrialFunction(U)
du = pt.TestFunction(U)
f = pt.interpolate(pt.Constant(1e-2), U)

class Left(pt.SubDomain):
    def inside(self, x, on_boundary):
        gamma = 2.0
        return x[0] == 0.0 and 50 - gamma < x[1] < 50 + gamma and on_boundary
bc = pt.DirichletBC(U, pt.Constant(0.0), Left())

design_variables = pt.DesignVariables()
design_variables.register(U,
                          "density",
                          [TARGET_DENSITY],
                          [(0, 1)],
                          lambda x: pt.helmholtz_filter(x, R=FILTER_RADIUS),
                          recording_path=output_path,
                          recording_interval=1)

class Problem(pt.ProblemStatement):
    def __init__(self):
        super().__init__()

    @pt.ProblemStatement.recording(self)
    def objective(self, design_variables):
        self.rho = design_variables["density"]
        a = pt.inner(pt.grad(u), simp(self.rho)*pt.grad(du)) * pt.dx
        L = pt.inner(f, du) * pt.dx
        pt.solve(a == L, uh, bc)
        return pt.assemble(pt.inner(f, uh) * pt.dx)
    
    def constraint_volume(self, design_variables):
        unitary = pt.project(pt.Constant(1), U)
        return pt.assemble(self.rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY
    

opt = pt.NloptOptimizer(design_variables, Problem(), "LD_MMA")
opt.set_maxeval(5)
opt.set_ftol_rel(1e-5)
opt.set_param("verbosity", 1)
opt.run(output_path + "/logging.csv")