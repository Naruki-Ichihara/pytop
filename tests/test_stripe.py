from pytop.toolkit.dehomogenization import sh_stripe
import pytop as pt
import matplotlib.pyplot as plt
mm = 1/(2.54*10)
plt.rcParams['figure.dpi'] = 1200
mpi = pt.MPI_Communicator.comm_world

dir = "output/stripe_test_small/"

# Define design domain
NUMBER_OF_NODES = (50, 50)
POSITION = (55, 55)
mesh = pt.RectangleMesh(pt.MPI_Communicator.comm_world, pt.Point(5, 5), pt.Point(*POSITION), *NUMBER_OF_NODES)
U = pt.FunctionSpace(mesh, 'CG', 1)
V = pt.VectorFunctionSpace(mesh, 'CG', 1)

rho = pt.create_initialized_fenics_function([1], U)
vec = pt.create_initialized_fenics_function([lambda x: x[0]/pt.sqrt(x[0]**2 + x[1]**2), lambda x: x[1]/pt.sqrt(x[0]**2 + x[1]**2)], V)
#vec_ = pt.project(pt.as_vector([-vec[1], vec[0]]), V)

stripe = sh_stripe(mesh, vec, rho, 1.0, times_of_mesh_refinement=2)
pt.save_fenics_function_to_file(mpi, stripe, dir + "stripe", "stripe", True)
file = pt.File(mpi, dir + "stripe.pvd")
file << stripe
"""
plt.figure(figsize=(POSITION[0]*mm, POSITION[1]*mm))
pt.plot(rho_, "density", cmap="binary", vmin=0, vmax=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('plot.tiff', bbox_inches='tight', pad_inches=0)
"""