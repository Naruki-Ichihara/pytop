import pytop as pt
from pytop.physics import elasticity as el
from pytop.physics.utils import penalized_weight, isoparametric_2D_box_to_triangle, sign

pt.parameters['form_compiler']['quadrature_degree'] = 5

# parameters
TARGET_DENSITY = 0.40
FILTER_RADIUS = 5.0
NUMBER_OF_NODES = (400, 200)
POSITION = (200, 100)
E1 = 1.e6
E2 = 1.e6/10
G12 = 1.e6/20
nu = 0.3
output_path = "output/orthro_elast_2D_cantilever_tensor"

# Mesh and function spaces
mesh = pt.RectangleMesh(pt.MPI_Communicator.comm_world, pt.Point(0, 0), pt.Point(*POSITION), *NUMBER_OF_NODES)
U = pt.VectorFunctionSpace(mesh, "CG", 1)
V = pt.VectorFunctionSpace(mesh, "CG", 1, dim=3)
X = pt.FunctionSpace(mesh, "CG", 1)
uh = pt.Function(U, name="displacement")
u = pt.TrialFunction(U)
du = pt.TestFunction(U)
f = pt.Constant((0, -1.0))

# Boundary conditions
class Left(pt.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 1e-6 and on_boundary
bc = pt.DirichletBC(U, pt.Constant((0, 0)), Left())
class Loading(pt.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 200-1e-6 and 45.0 < x[1] < 55.0 and on_boundary
    
ds = pt.make_noiman_boundary_domains(mesh, [Loading()], True)

# Design variables
def preprocess_for_density(x):
    x = pt.helmholtz_filter(x, R=FILTER_RADIUS)
    return x

def preprocess_for_orientation(x):
    x = pt.helmholtz_filter(x, R=FILTER_RADIUS)
    return x

def postprocess_for_density(x):
    x = pt.smooth_heviside_filter(x, beta=10.0, eta=0.5)
    return x

def postprocess_for_orientation(x):
    diagonals = isoparametric_2D_box_to_triangle(x[0], x[1])
    coupling_factor = diagonals[0]*diagonals[1]*sign(x[2])
    orientation_tensor_2 = pt.as_tensor([[diagonals[0], coupling_factor],
                                         [coupling_factor, diagonals[1]]])
    vector = pt.project(pt.as_vector([pt.sqrt(orientation_tensor_2[0, 0]), sign(x[2], 6)*pt.sqrt(orientation_tensor_2[1, 1])]), U)
    return vector

design_variables = pt.DesignVariables()
design_variables.register(X,
                          "density",
                          [TARGET_DENSITY],
                          [(0, 1)],
                          preprocess_for_density,
                          postprocess_for_density,
                          recording_path=output_path,
                          recording_interval=1)

design_variables.register(V,
                          "orientation",
                          [-0.99, -0.99, 0],
                          ([-1, 1], [-1, 1], [-1, 1]),
                          preprocess_for_orientation,
                          postprocess_for_orientation,
                          recording_path=output_path,
                          recording_interval=1)

# Problem statement
class Problem(pt.ProblemStatement):
    def objective(self, design_variables):
        self.rho = design_variables["density"]
        orientation_tensor_elems = design_variables["orientation"]
        diagonals = isoparametric_2D_box_to_triangle(orientation_tensor_elems[0], orientation_tensor_elems[1])
        coupling_factor = diagonals[0]*diagonals[1]*sign(orientation_tensor_elems[2])
        orientation_tensor_2 = pt.as_tensor([[diagonals[0], coupling_factor],
                                             [coupling_factor, diagonals[1]]])
        orientation_tensor_4 = pt.outer(orientation_tensor_2, orientation_tensor_2)
        a = el.linear_2D_orthotropic_elasticity_bilinear_form_tensor(u, du, E1, E2, G12, nu, orientation_tensor_2, orientation_tensor_4, penalized_weight(self.rho, eps=1e-4))
        L = pt.inner(f, du) * ds(1)
        pt.solve(a == L, uh, bc)
        return pt.assemble(pt.inner(f, uh) * ds(1))
    def constraint_volume(self, design_variables):
        unitary = pt.project(pt.Constant(1), X)
        return pt.assemble(self.rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY
        
    
# Optimization
opt = pt.NloptOptimizer(design_variables, Problem(), "LD_MMA")
opt.set_maxeval(10)
opt.set_ftol_rel(1e-5)
opt.set_param("verbosity", 1)
opt.run(output_path + "/logging.csv")