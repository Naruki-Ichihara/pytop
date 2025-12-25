import pytop as pt
from pytop.physics import elasticity as el
from pytop.physics.utils import penalized_weight, isoparametric_2D_box_to_circle

# parameters
TARGET_DENSITY = 0.25
FILTER_RADIUS = 1.0
NUMBER_OF_NODES = (100, 50)
E1 = 1.e6
E2 = 1.e6/10
G12 = 1.e6/20
nu = 0.3

# Mesh and function spaces
mesh = pt.RectangleMesh(pt.MPI_Communicator.comm_world, pt.Point(0, 0), pt.Point(20, 10), NUMBER_OF_NODES[0], NUMBER_OF_NODES[1])
U = pt.VectorFunctionSpace(mesh, "CG", 1)
X = pt.FunctionSpace(mesh, "CG", 1)
uh = pt.Function(U, name="displacement")
u = pt.TrialFunction(U)
du = pt.TestFunction(U)
f = pt.Constant((0, -1e2))

# Boundary conditions
class Left(pt.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 1e-6 and on_boundary
bc = pt.DirichletBC(U, pt.Constant((0, 0)), Left())
class Loading(pt.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 20-1e-6 and 4.5 < x[1] < 5.5 and on_boundary
    
ds = pt.make_noiman_boundary_domains(mesh, [Loading()], True)

# Design variables
def preprocess_for_density(x):
    x = pt.helmholtz_filter(x, R=FILTER_RADIUS)
    return x

def preprocess_for_orientation(x):
    x = pt.helmholtz_filter(x, R=FILTER_RADIUS)
    x = isoparametric_2D_box_to_circle(x[0], x[1])
    return pt.project(x, U)

design_variables = pt.DesignVariables()
design_variables.register(X,
                          "density",
                          [TARGET_DENSITY],
                          [(0, 1)],
                          preprocess_for_density,
                          recording_path="output/orthro_elast_2D_cantilever",
                          recording_interval=1)

design_variables.register(U,
                          "orientation",
                          [1, 0],
                          ([-1, 1], [-1, 1]),
                          preprocess_for_orientation,
                          recording_path="output/orthro_elast_2D_cantilever",
                          recording_interval=1)

# Problem statement
class Problem(pt.ProblemStatement):
    def objective(self, design_variables, iter_num):
        self.rho = design_variables["density"]
        self.orientation = design_variables["orientation"]
        a = el.linear_2D_orthotropic_elasticity_bilinear_form_vector(u, du, E1, E2, G12, nu, self.orientation, penalized_weight(self.rho, eps=1e-4))
        L = pt.inner(f, du) * ds(1)
        pt.solve(a == L, uh, bc)
        return pt.assemble(pt.inner(f, uh) * ds(1))
    def constraint_volume(self, design_variables, iter_num):
        unitary = pt.project(pt.Constant(1), X)
        return pt.assemble(self.rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY
    
# Optimization
opt = pt.NloptOptimizer(design_variables, Problem(), "LD_MMA")
opt.set_maxeval(100)
opt.set_ftol_rel(1e-4)
opt.set_param("verbosity", 1)
opt.run("output/orthro_elast_2D_cantilever/logging.csv")