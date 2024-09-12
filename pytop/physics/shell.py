from fenics import *
from fenics_adjoint import *
from ufl import RestrictedElement
from ufl import transpose, indices
from typing import Tuple, Callable

def z_coordinates(list_of_thickness: list) -> list:
    """Return a list with the thickness coordinate of the top surface of each layer
    taking the midplane as z = 0.
    Args:
        hs: a list giving the thinckesses of each layer
            ordered from bottom (layer - 0) to top (layer n-1).
    Returns:
        z: a list of coordinate of the top surface of each layer
           ordered from bottom (layer - 0) to top (layer n-1)
    """
    hs = list_of_thickness
    z = [(-sum(hs)/2. + sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)]
    return z

def nagdi_elements() -> MixedElement:
    """nagdi_elements. Return finite elements for the non linear nagdi shell.
    Args:
    Returns:
        elements: Mixed elements for nagdi shell.
                    
    """
    return MixedElement([VectorElement("Lagrange", triangle, 1, dim=3),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

def nagdi_strains(phi0: Function, d0: Function) -> Tuple[Callable, Callable, Callable]:
    """nagdi_strains. Return strain measures for the nagdi-shell model.
    Args:
        phi0: Reference configuration.
        d0: Reference director.
    Returns:
        e(F): Membrane strain measure.
        k(F, d): Bending strain measure.
        gamma(F, d): Shear strain measure.
    """
    a0 = grad(phi0).T*grad(phi0)
    b0 = -0.5*(grad(phi0).T*grad(d0) + grad(d0).T*grad(phi0))
    e = lambda F: 0.5*(F.T*F - a0)
    k = lambda F, d: -0.5*(F.T*grad(d)+grad(d).T*F) - b0
    gamma = lambda F, d: F.T*d - grad(phi0).T*d0
    return e, k, gamma

def AD_matrix_neo_hooken(mu: float, nu: float, list_of_thickness: list, index: list) -> Tuple[Form, Form]:
    """Return the stiffness matrix of an isotropic neo-hooken mechanical stiffness.
    Args:
        mu: shear stiffness for neo-hooken.
        nu: poisson's ratio.
        list_of_thickness: list of thickness of each layer.
        index: index of active layer.
    Returns:
        A: 
        D: 
    """

    z = z_coordinates(list_of_thickness)
    A = 0.
    D = 0.

    Y = 3*mu
    for i in range(len(list_of_thickness)):
        if i in index:
            A += Y/(1-nu**2)*(z[i+1]-z[i])
            D += Y/3/(1-nu**2)*(z[i+1]**3-z[i]**3)
        else:
            A += 0.
            D += 0.      

    return (A, D)

def dielectric_general_stiffness(relative_permittivity: float, list_of_thickness: list, index: int, eps0=8.85*1e-12):
    r"""Return the general stiffness matrix of an isotropic dielectric mechanics.
        epsr: Relative permittivity.
        list_of_thickness: a list with length n with the thicknesses of the layers (from top to bottom).
        h: total thickness.
        index: index of active layer.
        eps0: permittivity.
    Returns:
        A: general stiffness of membrane.
        D: general stiffness of bending.
    """
    z = z_coordinates(list_of_thickness)
    A = 0.
    D = 0.
    total_thickness = sum(list_of_thickness)
    for i in range(len(list_of_thickness)):
        if i in index:
            A += 4*eps0*relative_permittivity/total_thickness**2*(z[i+1]-z[i])
            D += 2*eps0*relative_permittivity/total_thickness**2*(z[i+1]**2-z[i]**2)
        else:
            A += 0.
            D += 0.
    D = D/A
    return (A, D)

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


def NM_T(E1, E2, G12, nu12, hs, thetas, index, DeltaT_0, DeltaT_1=0., alpha1=1., alpha2=1.):
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
        if i in index:
            N_T += Q_theta*alpha_theta*integral_DeltaT
            M_T += Q_theta*alpha_theta*integral_DeltaT_z
        else:
            pass

    return (N_T, M_T)

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