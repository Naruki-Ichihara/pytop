# Steady state Swift Hohenberg equation
from fenics import *
from fenics_adjoint import *
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
    
class GaussianRandomField_2D(UserExpression):
    def eval(self, val, x):
        val[0] = 10*np.random.randn()
        val[1] = 10*np.random.randn()
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

def sh_stripe(mesh: Mesh, source_vector: Function, source_density: Function, band_width: float, alpha=0.9, eps_0=1.0, g_0=0.0,
              absolute_tol=1e-2, max_iter=1000) -> Function:
    """Solve the steady state Swift-Hohenberg equation with stripe pattern.
    see: 
        https://doi.org/10.1038/s41598-023-41316-w
        https://doi.org/10.1016/j.compositesb.2022.109626

    Args:
        mesh: the mesh
        source_vector: the source vector
        source_density: the source density
        width: the width of the stripe
        alpha: the coefficient of the source term
        eps_0: the coefficient of the linear term
        g_0: the coefficient of the cubic term
        absolute_tol: the absolute tolerance
        max_iter: the maximum iterations

    Returns:
        stripe: the stripe pattern
    """
    width = band_width/source_density

    V = FiniteElement('CG', mesh.ufl_cell(), 2)
    M = FunctionSpace(mesh, V*V)
    X = FunctionSpace(mesh, 'CG', 1)

    theta = source_vector

    q = alpha*np.pi/width
    k = sqrt((np.pi/width)**2 - q**2)

    def G1(w, v):
        return -eps_0/2 - g_0/3*(w+v) + 1/4*(w**2+w*v+v**2)

    def G2(w):
        return -eps_0/2*w - g_0/3*w**2 + 1/4*w**3

    def A(w, v, k):
        return (dot(grad(w), grad(v)) - k**2*w*v)*source_density*dx

    def B(w, v, theta, q):
        D = outer(theta, theta)
        return + 2*q**2*dot(grad(w), dot(D, grad(v)))*source_density*dx

    Uh = Function(M)
    phi, psi = TestFunctions(M)

    initial = GaussianRandomField_2D()
    Uh.interpolate(initial)

    uh, qh = split(Uh)
    dPhi = G1(uh, uh)*uh + G2(uh)

    L0 = A(qh, phi, k) + dPhi*phi*dx - B(uh, phi, theta, q)
    L1 = qh*psi*dx - A(uh, psi, k)

    L = L0 + L1

    solve(L == 0, Uh, solver_parameters={"newton_solver":
                                        {"absolute_tolerance": absolute_tol,
                                         "maximum_iterations": max_iter}})
    
    stripe = project(Uh.split()[0], X)
    return stripe

def sh_stripe_tensor(mesh: Mesh, source_tensor: Function, source_density: Function, band_width: float, perpendicular=True, alpha=0.9, eps_0=1.0, g_0=0.0,
              absolute_tol=1e-2, max_iter=1000) -> Function:
    """Solve the steady state Swift-Hohenberg equation with stripe pattern.
    see: 
        https://doi.org/10.1038/s41598-023-41316-w
        https://doi.org/10.1016/j.compositesb.2022.109626

    Args:
        mesh: the mesh
        source_tensor: the source tensor (Assuming 3-dim vector function. 1. a11, 2. a22, and 3. a12)
        source_density: the source density
        band_width: the width of the stripe
        perpendicular: If True, the stripe is perpendicular to the source vector. Otherwise, the stripe is parallel to the source vector.
        alpha: the coefficient of the source term
        eps_0: the coefficient of the linear term
        g_0: the coefficient of the cubic term
        absolute_tol: the absolute tolerance
        max_iter: the maximum iterations

    Returns:
        stripe: the stripe pattern
    """
    width = band_width/source_density

    V = FiniteElement('CG', mesh.ufl_cell(), 2)
    M = FunctionSpace(mesh, V*V)
    X = FunctionSpace(mesh, 'CG', 1)

    if perpendicular:
        D = as_tensor([[source_tensor[1], -source_tensor[2]], [-source_tensor[2], source_tensor[0]]])
    else:
        D = as_tensor([[source_tensor[0], source_tensor[2]], [source_tensor[2], source_tensor[1]]])
    q = alpha*np.pi/width
    k = sqrt((np.pi/width)**2 - q**2)

    def G1(w, v):
        return -eps_0/2 - g_0/3*(w+v) + 1/4*(w**2+w*v+v**2)

    def G2(w):
        return -eps_0/2*w - g_0/3*w**2 + 1/4*w**3

    def A(w, v, k):
        return (dot(grad(w), grad(v)) - k**2*w*v)*source_density*dx

    def B(w, v, D, q):
        return + 2*q**2*dot(grad(w), dot(D, grad(v)))*source_density*dx

    Uh = Function(M)
    phi, psi = TestFunctions(M)

    initial = GaussianRandomField_2D()
    Uh.interpolate(initial)

    uh, qh = split(Uh)
    dPhi = G1(uh, uh)*uh + G2(uh)

    L0 = A(qh, phi, k) + dPhi*phi*dx - B(uh, phi, D, q)
    L1 = qh*psi*dx - A(uh, psi, k)

    L = L0 + L1

    solve(L == 0, Uh, solver_parameters={"newton_solver":
                                        {"absolute_tolerance": absolute_tol,
                                         "maximum_iterations": max_iter}})
    
    stripe = project(Uh.split()[0], X)
    return stripe