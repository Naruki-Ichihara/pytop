import pytop as pt
import pytop.physics.utils as utils

# parameters
TARGET_DENSITY = 0.3
FILTER_RADIUS = 0.05
NUMBER_OF_NODES = 100
output_path = "examples/output/poisson_problem"

mesh = pt.UnitSquareMesh(pt.MPI_Communicator.comm_world, NUMBER_OF_NODES, NUMBER_OF_NODES)
U = pt.FunctionSpace(mesh, "CG", 1)
uh = pt.Function(U)

u = pt.TrialFunction(U)
du = pt.TestFunction(U)
f = pt.interpolate(pt.Constant(1e-2), U)

class Left(pt.SubDomain):
    def inside(self, x, on_boundary):
        gamma = 0.1
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary
bc = pt.DirichletBC(U, pt.Constant(0.0), Left())

design_variables = pt.DesignVariables()
design_variables.register(U,
                          "density",
                          [TARGET_DENSITY],
                          [(0, 1)],
                          lambda x: pt.helmholtz_filter(x, R=FILTER_RADIUS),
                          recording_path=output_path,
                          recording_interval=1)

temp_file = pt.XDMFFile(output_path + "/temp.xdmf")

class Problem(pt.ProblemStatement):

    def objective(self, design_variables, iter_num):
        self.rho = design_variables["density"]
        a = pt.inner(pt.grad(u), utils.penalized_weight(self.rho)*pt.grad(du)) * pt.dx
        L = pt.inner(f, du) * pt.dx
        pt.solve(a == L, uh, bc)
        # Record the temperature field
        self.recorder(temp_file, uh*utils.penalized_weight(self.rho), U, "temp", iter_num)
        return pt.assemble(pt.inner(f, uh) * pt.dx)
    
    def constraint_volume(self, design_variables, iter_num):
        unitary = pt.project(pt.Constant(1), U)
        return pt.assemble(self.rho*pt.dx)/pt.assemble(unitary*pt.dx) - TARGET_DENSITY
    

opt = pt.NloptOptimizer(design_variables, Problem(), "LD_MMA")
opt.set_maxeval(100)
opt.set_ftol_rel(1e-5)
opt.set_param("verbosity", 1)
opt.run(output_path + "/logging.csv")