from fenics import *
from fenics_adjoint import *
from ufl import transpose, indices
import numpy as np
from typing import Callable, Iterable, Optional
from dataclasses import dataclass

def linear_2D_elasticity_bilinear_form(trial_function: TrialFunction, test_function: TestFunction, E: float, nu: float, weight: Callable=None) -> Form:
    '''Bilinear form of Linear 2D elasticity.

    Args: (TrialFunction, TestFunction, float, float, Callable)
        trial_function: trial function.
        test_function: test function.
        E: Young's modulus.
        nu: Poisson's ratio.
        weight: weight function.

    Returns: (Form)
        bilinear form.
    '''
    if weight is None:
        weight = Constant(1)
    a = inner(weight * elastic_2d_plane_stress(trial_function, E, nu), strain(test_function)) * dx
    return a

def linear_2D_orthotropic_elasticity_bilinear_form(trial_function: TrialFunction, test_function: TestFunction, E1: float, E2: float, G12: float, nu12: float, theta: float, weight: Callable=None) -> Form:
    '''Bilinear form of Linear 2D orthotropic elasticity.

    Args: (TrialFunction, TestFunction, float, float, float, float, float, Callable)
        trial_function: trial function.
        test_function: test function.
        E1: Young's modulus in the material direction 1.
        E2: Young's modulus in the material direction 2.
        G12: in-plane shear modulus.
        nu12: in-plane Poisson's ratio.
        theta: rotation angle from the material to the desired reference system.
        weight: weight function.

    Returns: (Form)
        bilinear form.
    '''
    if weight is None:
        weight = Constant(1)
    a = inner(weight * orthotropic_2d_plane_stress(trial_function, E1, E2, G12, nu12, theta), strain(test_function)) * dx
    return a

def linear_2D_orthotropic_elasticity_bilinear_form_vector(trial_function: TrialFunction, test_function: TestFunction, E1: float, E2: float, G12: float, nu12: float, orientation_vector: Function, weight: Callable=None) -> Form:
    '''Bilinear form of Linear 2D orthotropic elasticity.

    Args: (TrialFunction, TestFunction, float, float, float, float, Function, Callable)
        trial_function: trial function.
        test_function: test function.
        E1: Young's modulus in the material direction 1.
        E2: Young's modulus in the material direction 2.
        G12: in-plane shear modulus.
        nu12: in-plane Poisson's ratio.
        orientation_vector: orientation vector of the layer.
        weight: weight function.

    Returns: (Form)
        bilinear form.
    '''
    if weight is None:
        weight = Constant(1)
    a = inner(weight * orthotropic_2d_plane_stress_vector(trial_function, E1, E2, G12, nu12, orientation_vector), strain(test_function)) * dx
    return a

def linear_2D_orthotropic_elasticity_bilinear_form_tensor(trial_function: TrialFunction, test_function: TestFunction, E1: float, E2: float, G12: float, nu12: float, orient_tensor_2: Function, orient_tensor_4: Function, weight: Callable=None) -> Form:
    '''Bilinear form of Linear 2D orthotropic elasticity.

    Args: (TrialFunction, TestFunction, float, float, float, float, Function, Function, Callable)
        trial_function: trial function.
        test_function: test function.
        E1: Young's modulus in the material direction 1.
        E2: Young's modulus in the material direction 2.
        G12: in-plane shear modulus.
        nu12: in-plane Poisson's ratio.
        orient_tensor_2: orientation tensor of order 2.
        orient_tensor_4: orientation tensor of order 4.
        weight: weight function.

    Returns: (Form)
        bilinear form.
    '''
    if weight is None:
        weight = Constant(1)
    a = inner(weight * orthotropic_2d_plane_stress_tensor(trial_function, E1, E2, G12, nu12, orient_tensor_2, orient_tensor_4), strain(test_function)) * dx
    return a

def strain(u):
    '''Compute the strain tensor.
    
    Args: (Function)
        u: displacement field.
        
    Returns: (Function)
        strain tensor.    
    '''
    return sym(grad(u))

def elastic_2d_plane_stress(u, E, nu):
    '''Compute the elastic stress tensor.
    
    Args: (Function, float, float)
        u: displacement field.
        E: Young's modulus.
        nu: Poisson's ratio.
        
    Returns: (Function)
        stress tensor.    
    '''
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * strain(u) + lmbda * tr(strain(u)) * Identity(u.geometric_dimension())

def orthotropic_2d_plane_stress(u, E1, E2, G12, nu12, theta):
    '''Compute the orthotropic stress tensor.
    
    Args: (Function, float, float, float, float)
        u: displacement field.
        E1: Young's modulus in the material direction 1.
        E2: Young's modulus in the material direction 2.
        G12: in-plane shear modulus.
        nu12: in-plane Poisson ratio.
        theta: rotation angle from the material to the desired reference system.
        
    Returns: (Function)
        stress tensor.    
    '''
    Q = rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta)
    return stress_from_voigt(Q * strain_to_voigt(strain(u)))

def orthotropic_2d_plane_stress_vector(u, E1, E2, G12, nu12, orientation_vector):
    '''Compute the orthotropic stress tensor.
    
    Args: (Function, float, float, float, float)
        u: displacement field.
        E1: Young's modulus in the material direction 1.
        E2: Young's modulus in the material direction 2.
        G12: in-plane shear modulus.
        nu12: in-plane Poisson ratio.
        orientation_vector: orientation vector of the layer.
        
    Returns: (Function)
        stress tensor.    
    '''
    Q = rotated_lamina_stiffness_inplane_vector(E1, E2, G12, nu12, orientation_vector)
    return stress_from_voigt(Q * strain_to_voigt(strain(u)))

def orthotropic_2d_plane_stress_tensor(u, E1, E2, G12, nu12, orient_tensor_2, orient_tensor_4):
    '''Compute the orthotropic stress tensor.
    
    Args: (Function, float, float, float, float, Function, Function)
        u: displacement field.
        E1: Young's modulus in the material direction 1.
        E2: Young's modulus in the material direction 2.
        G12: in-plane shear modulus.
        nu12: in-plane Poisson ratio.
        orient_tensor_2: orientation tensor of order 2.
        orient_tensor_4: orientation tensor of order 4.
        
    Returns: (Function)
        stress tensor.    
    '''
    C = ortho_elast_2D_stiffness_tensor_from_orientation_tensor(orient_tensor_2, orient_tensor_4, E1, E2, G12, nu12)
    i, j, k, l = indices(4)
    return as_tensor(C[i, j, k, l]*strain(u)[k, l], (i, j))

def strain_to_voigt(e):
    r"""Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric strain tensor, according to the following rule (see e.g.
    https://en.wikipedia.org/wiki/Voigt_notation),
        .. math::
         e  = \begin{bmatrix} e_{00} & e_{01}\\ e_{01} & e_{11} \end{bmatrix}\quad\to\quad
         e_\mathrm{voigt}= \begin{bmatrix} e_{00} & e_{11}& 2e_{01} \end{bmatrix}
    Args:
        e: a symmetric 2x2 strain tensor, typically UFL form with shape (2,2)
    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt
        notation.
    """
    return as_vector((e[0, 0], e[1, 1], 2*e[0, 1]))

def stress_to_voigt(sigma):
    r"""Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric stress tensor, according to the following rule (see e.g.
    https://en.wikipedia.org/wiki/Voigt_notation),
        .. math::
         \sigma  = \begin{bmatrix} \sigma_{00} & \sigma_{01}\\ \sigma_{01} & \sigma_{11} \end{bmatrix}\quad\to\quad
         \sigma_\mathrm{voigt}= \begin{bmatrix} \sigma_{00} & \sigma_{11}& \sigma_{01} \end{bmatrix}
    Args:
        sigma: a symmetric 2x2 stress tensor, typically UFL form with shape
        (2,2).
    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt notation.
    """
    return as_vector((sigma[0, 0], sigma[1, 1], sigma[0, 1]))

def strain_from_voigt(e_voigt):
    r"""Inverse operation of strain_to_voigt.
    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the strain
        pseudo-vector in Voigt format
    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((e_voigt[0], e_voigt[2]/2.), (e_voigt[2]/2., e_voigt[1])))


def stress_from_voigt(sigma_voigt):
    r"""Inverse operation of stress_to_voigt.
    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the stress
        pseudo-vector in Voigt format.
    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))

def z_coordinates(hs):
    r"""Return a list with the thickness coordinate of the top surface of each layer
    taking the midplane as z = 0.
    Args:
        hs: a list giving the thinckesses of each layer
            ordered from bottom (layer - 0) to top (layer n-1).
    Returns:
        z: a list of coordinate of the top surface of each layer
           ordered from bottom (layer - 0) to top (layer n-1)
    """

    z0 = sum(hs)/2.
    z = [(-sum(hs)/2. + sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)]
    return z


def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    r"""Return the in-plane stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (see Reddy 1997, eqn 1.3.71)
    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G23: The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        theta: The rotation angle from the material to the desired reference system.
    Returns:
        Q_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix.
    """
    # Rotation matrix to rotate the in-plane stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)

    c = cos(theta)
    s = sin(theta)
    T = as_matrix([[c**2, s**2, -2*s*c],
                   [s**2, c**2, 2*s*c],
                   [c*s, -c*s, c**2 - s**2]])
    # In-plane stiffness matrix of an orhtropic layer in material coordinates
    nu21 = E2/E1*nu12
    Q11 = E1/(1-nu12*nu21)
    Q12 = nu12*E2/(1-nu12*nu21)
    Q22 = E2/(1-nu12*nu21)
    Q66 = G12
    Q16 = 0.
    Q26 = 0.
    Q = as_matrix([[Q11, Q12, Q16],
                   [Q12, Q22, Q26],
                   [Q16, Q26, Q66]])
    # Rotated matrix in the main reference
    Q_theta = T*Q*transpose(T)

    return Q_theta


def rotated_lamina_stiffness_shear(G13, G23, theta, kappa=5./6.):
    r"""Return the shear stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state
    (see Reddy 1997, eqn 3.4.18).
    Args:
        G12: The transverse shear modulus between the material directions 1-2.
        G13: The transverse shear modulus between the material directions 1-3.
        kappa: The shear correction factor.
    Returns:
        Q_shear_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix.
    """
    # The rotation matrix to rotate the shear stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)
    c = cos(theta)
    s = sin(theta)
    T_shear = as_matrix([[c, s], [-s, c]])
    Q_shear = kappa*as_matrix([[G23, 0.], [0., G13]])
    Q_shear_theta = T_shear*Q_shear*transpose(T_shear)

    return Q_shear_theta


def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    r"""Return the in-plane stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (See Reddy 1997, eqn 1.3.71)
    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G23: The in-plane shear modulus
        nu12: The in-plane Poisson ratio
        theta: The rotation angle from the material to the desired refence system
    Returns:
        Q_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix
    """
    # Rotation matrix to rotate the in-plane stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)

    c = cos(theta)
    s = sin(theta)
    T = as_matrix([[c**2, s**2, -2*s*c],
                   [s**2, c**2, 2*s*c],
                   [c*s, -c*s, c**2-s**2]])

    # In-plane stiffness matrix of an orhtropic layer in material coordinates
    nu21 = E2/E1*nu12
    Q11 = E1/(1-nu12*nu21)
    Q12 = nu12*E2/(1-nu12*nu21)
    Q22 = E2/(1-nu12*nu21)
    Q66 = G12
    Q16 = 0.
    Q26 = 0.
    Q = as_matrix([[Q11, Q12, Q16],
                   [Q12, Q22, Q26],
                   [Q16, Q26, Q66]])
    # Rotated matrix in the main reference
    Q_theta = T*Q*transpose(T)

    return Q_theta

def rotated_lamina_stiffness_inplane_vector(E1, E2, G12, nu12, orientation_vector):
    r"""Return the in-plane stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (see Reddy 1997, eqn 1.3.71)
    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G23: The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        orientation_vector: The orientation vector of the layer.
        
    Returns:
        Q_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix.
    """
    # Rotation matrix to rotate the in-plane stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)

    c = orientation_vector[0]
    s = orientation_vector[1]
    T = as_matrix([[c**2, s**2, -2*s*c],
                   [s**2, c**2, 2*s*c],
                   [c*s, -c*s, c**2 - s**2]])
    # In-plane stiffness matrix of an orhtropic layer in material coordinates
    nu21 = E2/E1*nu12
    Q11 = E1/(1-nu12*nu21)
    Q12 = nu12*E2/(1-nu12*nu21)
    Q22 = E2/(1-nu12*nu21)
    Q66 = G12
    Q16 = 0.
    Q26 = 0.
    Q = as_matrix([[Q11, Q12, Q16],
                   [Q12, Q22, Q26],
                   [Q16, Q26, Q66]])
    # Rotated matrix in the main reference
    Q_theta = T*Q*transpose(T)

    return Q_theta


def ABD(E1, E2, G12, nu12, hs_grobal, thetas_grobal, index):
    r"""Return the stiffness matrix of a kirchhoff-love model of a laminate
    obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations (see Reddy 1997, eqn 1.3.71).
    It assumes a plane-stress state.
    Args:
        E1 : The Young modulus in the material direction 1.
        E2 : The Young modulus in the material direction 2.
        G12 : The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        hs: a list with length n with the thicknesses of the layers (from top to bottom).
        theta: a list with the n orientations (in radians) of the layers (from top to bottom).
    Returns:
        A: a symmetric 3x3 ufl matrix giving the membrane stiffness in Voigt notation.
        B: a symmetric 3x3 ufl matrix giving the membrane/bending coupling stiffness in Voigt notation.
        D: a symmetric 3x3 ufl matrix giving the bending stiffness in Voigt notation.
    """
    assert (len(hs_grobal) == len(thetas_grobal)), "hs and thetas should have the same length !"

    z = z_coordinates(hs_grobal)
    A = 0.*Identity(3)
    B = 0.*Identity(3)
    D = 0.*Identity(3)

    for i in range(len(hs_grobal)):
        if i in index:
            Qbar = rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, thetas_grobal[i])
            A += Qbar*(z[i+1]-z[i])
            B += .5*Qbar*(z[i+1]**2-z[i]**2)
            D += 1./3.*Qbar*(z[i+1]**3-z[i]**3)
        else:
            pass
    return (A, B, D)


def Fs(G13, G23, hs_grobal, thetas_grobal, index):
    r"""Return the shear stiffness matrix of a Reissner-Midlin model of a
    laminate obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations.  (See Reddy 1997, eqn 3.4.18)
    It assumes a plane-stress state.
    Args:
        G13: The transverse shear modulus between the material directions 1-3.
        G23: The transverse shear modulus between the material directions 2-3.
        hs: a list with length n with the thicknesses of the layers (from top to bottom).
        theta: a list with the n orientations (in radians) of the layers (from top to bottom).
    Returns:
        F: a symmetric 2x2 ufl matrix giving the shear stiffness in Voigt notation.
    """
    assert (len(hs_grobal) == len(thetas_grobal)), "hs and thetas should have the same length !"

    z = z_coordinates(hs_grobal)
    F = 0.*Identity(2)

    for i in range(len(hs_grobal)):
        if i in index:
            Q_shear_theta = rotated_lamina_stiffness_shear(G13, G23, thetas_grobal[i])
            F += Q_shear_theta*(z[i+1]-z[i])
        else:
            pass

    return F


def rotated_lamina_expansion_inplane(alpha11, alpha22, theta):
    r"""Return the in-plane expansion matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (See Reddy 1997, eqn 1.3.71)
    Args:
        alpha11: Expansion coefficient in the material direction 1.
        alpha22: Expansion coefficient in the material direction 2.
        theta: The rotation angle from the material to the desired reference system.
    Returns:
        alpha_theta: a 3x1 ufl vector giving the expansion matrix in voigt notation.
    """
    # Rotated matrix, assuming alpha12 = 0
    c = cos(theta)
    s = sin(theta)
    alpha_xx = alpha11*c**2 + alpha22*s**2
    alpha_yy = alpha11*s**2 + alpha22*c**2
    alpha_xy = 2*(alpha11-alpha22)*s*c
    alpha_theta = as_vector([alpha_xx, alpha_yy, alpha_xy])

    return alpha_theta


def NM_T(E1, E2, G12, nu12, hs, thetas, DeltaT_0, DeltaT_1=0., alpha1=1., alpha2=1.):
    r"""Return the thermal stress and moment resultant of a Kirchhoff-Love model
    of a laminate obtained by stacking n orthotropic laminae with possibly
    different thinknesses and orientations.
    It assumes a plane-stress states and a temperature distribution in the from
    Delta(z) = DeltaT_0 + z * DeltaT_1
    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G12: The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        hs: a list with length n with the thicknesses of the layers (from top to bottom).
        theta: a list with the n orientations (in radians) of the layers (from top to bottom).
        alpha1: Expansion coefficient in the material direction 1.
        alpha2: Expansion coefficient in the material direction 2.
        DeltaT_0: Average temperature field.
        DeltaT_1: Gradient of the temperature field.
    Returns:
        N_T: a 3x1 ufl vector giving the membrane inelastic stress.
        M_T: a 3x1 ufl vector giving the bending inelastic stress.
    """
    assert (len(hs) == len(thetas)), "hs and thetas should have the same length !"
    # Coordinates of the interfaces
    z = z_coordinates(hs)

    # Initialize to zero the voigt (ufl) vectors
    N_T = as_vector((0., 0., 0.))
    M_T = as_vector((0., 0., 0.))

    T0 = DeltaT_0
    T1 = DeltaT_1

    # loop over the layers to add the different contributions
    for i in range(len(thetas)):
        # Rotated stiffness
        Q_theta = rotated_lamina_stiffness_inplane(
            E1, E2, G12, nu12, thetas[i])
        alpha_theta = rotated_lamina_expansion_inplane(
            alpha1, alpha2, thetas[i])
        # numerical integration in the i-th layer
        z0i = (z[i+1] + z[i])/2  # Midplane of the ith layer
        # integral of DeltaT(z) in (z[i+1], z[i])
        integral_DeltaT = hs[i]*(T0 + T1 * z0i)
        # integral of DeltaT(z)*z in (z[i+1], z[i])
        integral_DeltaT_z = T1*(hs[i]*3/12 + z0i**2*hs[i]) + hs[i]*z0i*T0
        N_T += Q_theta*alpha_theta*integral_DeltaT
        M_T += Q_theta*alpha_theta*integral_DeltaT_z

    return (N_T, M_T)

def ortho_elast_2D_stiffness_tensor_from_orientation_tensor(orient_tensor_2: Function, orient_tensor_4: Function, E1: float, E2: float, G12: float, nu12: float) -> Function:
    ''' Compute the stiffness tensor of an orthotropic material from the orientation tensor.

    Args: (Function, Function, float, float, float, float)
        orient_tensor_2: orientation tensor of order 2.
        orient_tensor_4: orientation tensor of order 4.
        E1: Young's modulus in the material direction 1.
        E2: Young's modulus in the material direction 2.
        G12: in-plane shear modulus.
        nu12: in-plane Poisson ratio.

    Returns: (Function)
        stiffness tensor.
    '''

    a_2 = orient_tensor_2
    a_4 = orient_tensor_4
    i, j, k, l = indices(4)
    nu21 = nu12*E2/E1
    m = 1/(1-nu12*nu21)

    C1111 = m*E1
    C1122 = m*nu21*E1
    C2222 = m*E2
    C1212 = G12

    B1 = C1111 + C2222 -2*C1122 - 4*C1212
    B2 = C1122
    B3 = C1212 - C2222/2
    B4 = 0
    B5 = C2222/2

    delta = Identity(2)
    c1 = B1*a_4[i, j, k, l]
    c2 = B2*(a_2[i, j]*delta[k, l] + a_2[k, l]*delta[i, j])
    c3 = B3*(a_2[i, k]*delta[j, l]+a_2[i, l]*delta[j, k] + a_2[j, k]*delta[i, l]+a_2[j, l]*delta[i, k])
    c4 = 0
    c5 = B5*(delta[i, k]*delta[j, l] + delta[i, l]*delta[j, k])

    C = as_tensor(c1+c2+c3+c5, (i, j, k, l))
    return C