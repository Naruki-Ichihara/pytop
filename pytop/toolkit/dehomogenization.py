# Steady state Swift Hohenberg equation
from fenics import *
from fenics_adjoint import *
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
    
class GaussianRandomField_2D(UserExpression):
    def eval(self, val, x):
        val[0] = 1*np.random.randn()
        val[1] = 1*np.random.randn()
    def value_shape(self):
        return (2,)

class Problem(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

def sh_stripe(mesh, source_vector, alpha, eps_0, g_0, w_0, rho, times_of_mesh_refinement=1, initial_noise_coffcient=10):

    mesh_fine = mesh
    for i in range(times_of_mesh_refinement):
        mesh_fine = refine(mesh_fine)

    V = FiniteElement('CG', mesh.ufl_cell(), 2)
    M = FunctionSpace(mesh, V*V)
    X = FunctionSpace(mesh_fine, 'CG', 1)

    theta = source_vector

    q = alpha*np.pi/w_0
    k = sqrt((np.pi/w_0)**2 - q**2)

    def G1(w, v):
        return -eps_0/2 - g_0/3*(w+v) + 1/4*(w**2+w*v+v**2)

    def G2(w):
        return -eps_0/2*w - g_0/3*w**2 + 1/4*w**3

    def A(w, v, k):
        return (dot(grad(w), grad(v)) - k**2*w*v)*rho*dx

    def B(w, v, theta, q):
        D = outer(theta, theta)
        return + 2*q**2*dot(grad(w), dot(D, grad(v)))*rho*dx

    Uh = Function(M)
    U = TrialFunction(M)
    phi, psi = TestFunctions(M)

    initial = GaussianRandomField_2D()
    Uh.interpolate(initial)

    uh, qh = split(Uh)
    dPhi = G1(uh, uh)*uh + G2(uh)

    L0 = A(qh, phi, k) + dPhi*phi*dx - B(uh, phi, theta, q)
    L1 = qh*psi*dx - A(uh, psi, k)

    L = L0 + L1

    solve(L == 0, Uh, solver_parameters={"newton_solver":
                                    {"absolute_tolerance": 1e-2,
                                     "maximum_iterations": 1000}})
    
    stripe = project(Uh.split()[0], X)
    return stripe