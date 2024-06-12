from pytop.toolkit.dehomogenization import sh_stripe
import pytop as pt

mpi = pt.MPI_Communicator.comm_world

dir = "output/stripe_test/"

# Define design domain
L = 5
H = 5
d = 0.5
mesh = pt.RectangleMesh(mpi, pt.Point((10, 10)), pt.Point((L+10, H+10)), int(L/d), int(H/d))
U = pt.FunctionSpace(mesh, 'CG', 1)
V = pt.VectorFunctionSpace(mesh, 'CG', 1)

rho = pt.create_initialized_fenics_function([1], U)
vec = pt.create_initialized_fenics_function([lambda x: x[0]/pt.sqrt(x[0]**2 + x[1]**2), lambda x: x[1]/pt.sqrt(x[0]**2 + x[1]**2)], V)

stripe = pt.read_fenics_function_from_file(dir + "stripe", U)
#stripe = sh_stripe(mesh, vec, rho, 1.0, times_of_mesh_refinement=0)
pt.save_fenics_function_to_file(mpi, stripe, dir + "stripe", "stripe", True)