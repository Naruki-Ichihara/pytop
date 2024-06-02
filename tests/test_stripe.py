from pytop.toolkit.dehomogenization import sh_stripe
import pytop as pt

alpha = 0.9
eps_0 = 1.0
g_0 = 0
w0 = 1.0

dir = "output/stripe_test/"

# Define design domain
L = 50
H = 50
d = 0.5
mesh = pt.RectangleMesh(pt.Point((10, 10)), pt.Point((L+10, H+10)), int(L/d), int(H/d))
U = pt.FunctionSpace(mesh, 'CG', 1)
V = pt.VectorFunctionSpace(mesh, 'CG', 1)

rho = pt.create_initialized_fenics_function([1], U)
vec = pt.create_initialized_fenics_function([lambda x: x[0]/pt.sqrt(x[0]**2 + x[1]**2), lambda x: x[1]/pt.sqrt(x[0]**2 + x[1]**2)], V)

w = w0/rho

stripe = sh_stripe(mesh, vec, alpha, eps_0, g_0, w, rho, 2)
file = pt.File(dir + 'test_.pvd')
file << stripe