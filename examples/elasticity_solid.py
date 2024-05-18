import pytop as pt
from pytop.physics import elasticity as mech

# parameters
TARGET_DENSITY = 0.25
FILTER_RADIUS = 1.0
NUMBER_OF_NODES = (150, 30)
E = 1.e6
nu = 0.3

mesh = pt.RectangleMesh(pt.MPI_Communicator.comm_world, pt.Point(0, 0), pt.Point(50, 10), NUMBER_OF_NODES[0], NUMBER_OF_NODES[1])
U = pt.VectorFunctionSpace(mesh, "CG", 1)
X = pt.FunctionSpace(mesh, "CG", 1)
uh = pt.Function(U, name="displacement")
u = pt.TrialFunction(U)
du = pt.TestFunction(U)
f = pt.Constant((0, -1))

class Left(pt.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 1e-6 and on_boundary
bc = pt.DirichletBC(U, pt.Constant((0, 0)), Left())

class Loading(pt.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 50-1e-6 and 4.8 < x[1] < 5.2 and on_boundary
    
ds = pt.make_noiman_boundary_domains(mesh, [Loading()])

design_variables = pt.DesignVariables()
design_variables.register(X,
                          "density",
                          [0.5],
                          [(0, 1)],
                          lambda x: pt.helmholtz_filter(x, R=FILTER_RADIUS),
                          recording_path="output/elasticity_solid",
                          recording_interval=10)

class Problem(pt.ProblemStatement):
    def objective(self, design_variables):
        self.rho = design_variables["density"]
        a = mech.linear_2D_elasticity_bilinear_form(u, du, E, nu, mech.penalized_weight(self.rho, eps=1e-2))
        L = pt.inner(f, du) * ds
        pt.solve(a == L, uh, bc)
        return pt.assemble(pt.inner(f, uh) * ds)
    def constraint_volume(self, design_variables):
        unitary = pt.project(pt.Constant(1), X)
        return pt.assemble(self.rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY

opt = pt.NloptOptimizer(design_variables, Problem(), "LD_MMA")
opt.set_maxeval(300)
opt.set_ftol_rel(1e-5)
opt.set_param("verbosity", 1)
opt.run("output/elasticity_solid/logging.csv")