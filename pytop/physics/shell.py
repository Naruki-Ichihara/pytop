from fenics import *
from fenics_adjoint import *
from ufl import RestrictedElement
from ufl import transpose, indices
from typing import Tuple, Callable

def z_coordinates(list_of_thickness: list) -> list:
    """Return a list with the thickness coordinate of the top surface of each layer
    taking the midplane as z = 0.
    Args:
        list_of_thickness: a list giving the thinckesses of each layer
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
        e: Membrane strain measure.
        k: Bending strain measure.
        gamma: Shear strain measure.
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
        A: Stiffness matrix of membrane.
        D: Stiffness matrix of bending.
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

def ABD(E1: float | Function,
        E2: float | Function,
        G12: float | Function,
        nu12: float | Function,
        list_of_thickness: list[float | Function],
        orientations: list[float | Function],
        index : list[int]) -> Tuple[Form, Form, Form]:
    r"""Return the stiffness matrix of a kirchhoff-love model of a laminate
    obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations (see Reddy 1997, eqn 1.3.71).
    It assumes a plane-stress state.
    Args:
        E1 : The Young modulus in the material direction 1.
        E2 : The Young modulus in the material direction 2.
        G12 : The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        list_of_thickness: a list with length n with the thicknesses of the layers (from top to bottom).
        orientations: a list with the n orientations (in radians) of the layers (from top to bottom).
        index: index of active layer.
    Returns:
        A: a symmetric 3x3 ufl matrix giving the membrane stiffness in Voigt notation.
        B: a symmetric 3x3 ufl matrix giving the membrane/bending coupling stiffness in Voigt notation.
        D: a symmetric 3x3 ufl matrix giving the bending stiffness in Voigt notation.
    """
    assert (len(list_of_thickness) == len(orientations)), "hs and thetas should have the same length !"

    z = z_coordinates(list_of_thickness)
    A = 0.*Identity(3)
    B = 0.*Identity(3)
    D = 0.*Identity(3)

    for i in range(len(list_of_thickness)):
        if i in index:
            Qbar = rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, orientations[i])
            A += Qbar*(z[i+1]-z[i])
            B += .5*Qbar*(z[i+1]**2-z[i]**2)
            D += 1./3.*Qbar*(z[i+1]**3-z[i]**3)
        else:
            pass
    return (A, B, D)

def ABD_tensor(E1: float | Function,
               E2: float | Function,
               G12: float | Function,
               nu12: float | Function,
               list_of_thickness: list[float | Function],
               orientation_tensors: list[Function | Form],
               index : list[int]) -> Tuple[Form, Form, Form]:
    r"""Return the stiffness matrix of a kirchhoff-love model of a laminate
    obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations (see Reddy 1997, eqn 1.3.71).
    It assumes a plane-stress state.
    Args:
        E1 : The Young modulus in the material direction 1.
        E2 : The Young modulus in the material direction 2.
        G12 : The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        list_of_thickness: a list with length n with the thicknesses of the layers (from top to bottom).
        orientation_tensors: a list with the n orientations (in radians) of the layers (from top to bottom).
        index: index of active layer.
    Returns:
        A: a symmetric 3x3 ufl matrix giving the membrane stiffness in Voigt notation.
        B: a symmetric 3x3 ufl matrix giving the membrane/bending coupling stiffness in Voigt notation.
        D: a symmetric 3x3 ufl matrix giving the bending stiffness in Voigt notation.
    """

    assert (len(list_of_thickness) == len(orientation_tensors)), "hs and thetas should have the same length !"

    z = z_coordinates(list_of_thickness)
    A = 0.*Identity(3)
    B = 0.*Identity(3)
    D = 0.*Identity(3)

    for i in range(len(list_of_thickness)):
        if i in index:
            Qbar = rotated_lamina_stiffness_inplane_tensor(E1, E2, G12, nu12, orientation_tensors[i])
            A += Qbar*(z[i+1]-z[i])
            B += .5*Qbar*(z[i+1]**2-z[i]**2)
            D += 1./3.*Qbar*(z[i+1]**3-z[i]**3)
        else:
            pass
    return (A, B, D)


def Fs(G13: float | Function,
       G23: float | Function,
       list_of_thickness: list[float | Function],
       orientations: list[float | Function],
       index: list[int]) -> Form:
    r"""Return the shear stiffness matrix of a Reissner-Midlin model of a
    laminate obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations.  (See Reddy 1997, eqn 3.4.18)
    It assumes a plane-stress state.
    Args:
        G13: The transverse shear modulus between the material directions 1-3.
        G23: The transverse shear modulus between the material directions 2-3.
        list_of_thickness: a list with length n with the thicknesses of the layers (from top to bottom).
        orientations: a list with the n orientations (in radians) of the layers (from top to bottom).
        index: index of active layer.
    Returns:
        F: a symmetric 2x2 ufl matrix giving the shear stiffness in Voigt notation.
    """
    assert (len(list_of_thickness) == len(orientations)), "hs and thetas should have the same length !"

    z = z_coordinates(list_of_thickness)
    F = 0.*Identity(2)

    for i in range(len(list_of_thickness)):
        if i in index:
            Q_shear_theta = rotated_lamina_stiffness_shear(G13, G23, orientations[i])
            F += Q_shear_theta*(z[i+1]-z[i])
        else:
            pass

    return F

def Fs_tensor(G13: float | Function,
              G23: float | Function,
              list_of_thickness: float | Function,
              orientation_tensors: list[Function | Form],
              index: list[int]) -> Form:
    r"""Return the shear stiffness matrix of a Reissner-Midlin model of a
    laminate obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations.  (See Reddy 1997, eqn 3.4.18)
    It assumes a plane-stress state.
    Args:
        G13: The transverse shear modulus between the material directions 1-3.
        G23: The transverse shear modulus between the material directions 2-3.
        list_of_thickness: a list with length n with the thicknesses of the layers (from top to bottom).
        orientation_tensors: a list with the n orientations (in radians) of the layers (from top to bottom).
    Returns:
        F: a symmetric 2x2 ufl matrix giving the shear stiffness in Voigt notation.
    """
    assert (len(list_of_thickness) == len(orientation_tensors)), "hs and thetas should have the same length !"

    z = z_coordinates(list_of_thickness)
    F = 0.*Identity(2)

    for i in range(len(list_of_thickness)):
        if i in index:
            Q_shear_theta = rotated_lamina_stiffness_shear_tensor(G13, G23, orientation_tensors[i])
            F += Q_shear_theta*(z[i+1]-z[i])
        else:
            pass

    return F

def rotated_lamina_expansion_inplane(alpha11: float, alpha22: float, theta: float) -> Form:
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
    c = cos(theta)
    s = sin(theta)
    alpha_xx = alpha11*c**2 + alpha22*s**2
    alpha_yy = alpha11*s**2 + alpha22*c**2
    alpha_xy = 2*(alpha11-alpha22)*s*c
    alpha_theta = as_vector([alpha_xx, alpha_yy, alpha_xy])

    return alpha_theta

def rotated_lamina_stiffness_shear(G13: float, G23: float, theta: float, kappa=5./6.) -> Form:
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

def rotated_lamina_stiffness_shear_tensor(G13: float, G23: float, tensor: Function | Form, kappa=5./6.) -> Form:
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
    a11 = tensor[0, 0]
    a22 = tensor[1, 1]
    a12 = tensor[0, 1]
    G_xx = a11*(G23-G13) + G13
    G_yy = a22*(G13-G23) + G23
    G_xy = 2*a12*(G23-G13)
    Q_shear_theta = kappa*as_matrix([[G_xx, G_xy], [G_xy, G_yy]])

    return Q_shear_theta

def rotated_lamina_expansion_inplane_tensor(alpha11: float, alpha22: float, orientation_tensors: Function | Form) -> Form:
    r"""Return the in-plane expansion matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (See Reddy 1997, eqn 1.3.71)
    Args:
        alpha11: Expansion coefficient in the material direction 1.
        alpha22: Expansion coefficient in the material direction 2.
        orientation_tensors: a 2x2 ufl matrix giving the orientation matrix.
    Returns:
        alpha_theta: a 3x1 ufl vector giving the expansion matrix in voigt notation.
    """
    a11 = orientation_tensors[0, 0]
    a22 = orientation_tensors[1, 1]
    a12 = orientation_tensors[0, 1]
    alpha_xx = a11*(alpha11-alpha22) + alpha22
    alpha_yy = a22*(alpha22-alpha11) + alpha11
    alpha_xy = 2*a12*(alpha11-alpha22)
    alpha_theta = as_vector([alpha_xx, alpha_yy, alpha_xy])

    return alpha_theta

def rotated_lamina_stiffness_inplane_tensor(E1: float | Function,
                                            E2: float | Function,
                                            G12: float | Function,
                                            nu12: float | Function,
                                            orientation_tensors: Function | Form) -> Form:
    r"""Return the in-plane stiffness matrix of an orhtotropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (See Reddy 1997, eqn 1.3.71)
    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G23: The in-plane shear modulus
        nu12: The in-plane Poisson ratio
        orientation_tensors: a 2x2 ufl matrix giving the orientation matrix.
    Returns:
        Q_voigt: a 3x3 symmetric ufl matrix giving the stiffness matrix
    """
    a_2 = orientation_tensors
    a_4 = outer(orientation_tensors, orientation_tensors)
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
    B5 = C2222/2

    delta = Identity(2)
    c1 = B1*a_4[i, j, k, l]
    c2 = B2*(a_2[i, j]*delta[k, l] + a_2[k, l]*delta[i, j])
    c3 = B3*(a_2[i, k]*delta[j, l]+a_2[i, l]*delta[j, k] + a_2[j, k]*delta[i, l]+a_2[j, l]*delta[i, k])
    c5 = B5*(delta[i, k]*delta[j, l] + delta[i, l]*delta[j, k])

    Q = as_tensor(c1+c2+c3+c5, (i, j, k, l))
    Q_voigt = as_matrix([[Q[0, 0, 0, 0], Q[0, 0, 1, 1], Q[0, 0, 0, 1]],
                            [Q[1, 1, 0, 0], Q[1, 1, 1, 1], Q[1, 1, 0, 1]],
                            [Q[0, 1, 0, 0], Q[0, 1, 1, 1], Q[0, 1, 0, 1]]])
    return Q_voigt

def NM_T_tensor(E1, E2, G12, nu12, hs, orientation_tensors, index, DeltaT_0, DeltaT_1=0., alpha1=1., alpha2=1.):
    assert (len(hs) == len(orientation_tensors)), "hs and thetas should have the same length !"
    # Coordinates of the interfaces
    z = z_coordinates(hs)

    # Initialize to zero the voigt (ufl) vectors
    N_T = as_vector((0., 0., 0.))
    M_T = as_vector((0., 0., 0.))

    T0 = DeltaT_0
    T1 = DeltaT_1

    # loop over the layers to add the different contributions
    for i in range(len(orientation_tensors)):
        # Rotated stiffness
        Q_theta = rotated_lamina_stiffness_inplane_tensor(
            E1, E2, G12, nu12, orientation_tensors[i])
        alpha_theta = rotated_lamina_expansion_inplane_tensor(
            alpha1, alpha2, orientation_tensors[i])
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


def NM_T(E1: float,
         E2: float,
         G12: float,
         nu12: float,
         list_of_thickness: list[float],
         orientations: list[float],
         index: list[int],
         DeltaT_0: float,
         DeltaT_1: float,
         alpha1 : float,
         alpha2 : float) -> Tuple[Form, Form]:
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
        list_of_thickness: a list with length n with the thicknesses of the layers (from top to bottom).
        orientations: a list with the n orientations (in radians) of the layers (from top to bottom).
        alpha1: Expansion coefficient in the material direction 1.
        alpha2: Expansion coefficient in the material direction 2.
        DeltaT_0: Average temperature field.
        DeltaT_1: Gradient of the temperature field.
    Returns:
        N_T: a 3x1 ufl vector giving the membrane inelastic stress.
        M_T: a 3x1 ufl vector giving the bending inelastic stress.
    """
    assert (len(list_of_thickness) == len(orientations)), "hs and thetas should have the same length !"
    # Coordinates of the interfaces
    z = z_coordinates(list_of_thickness)

    # Initialize to zero the voigt (ufl) vectors
    N_T = as_vector((0., 0., 0.))
    M_T = as_vector((0., 0., 0.))

    T0 = DeltaT_0
    T1 = DeltaT_1

    # loop over the layers to add the different contributions
    for i in range(len(orientations)):
        # Rotated stiffness
        Q_theta = rotated_lamina_stiffness_inplane(
            E1, E2, G12, nu12, orientations[i])
        alpha_theta = rotated_lamina_expansion_inplane(
            alpha1, alpha2, orientations[i])
        # numerical integration in the i-th layer
        z0i = (z[i+1] + z[i])/2  # Midplane of the ith layer
        # integral of DeltaT(z) in (z[i+1], z[i])
        integral_DeltaT = list_of_thickness[i]*(T0 + T1 * z0i)
        # integral of DeltaT(z)*z in (z[i+1], z[i])
        integral_DeltaT_z = T1*(list_of_thickness[i]*3/12 + z0i**2*list_of_thickness[i]) + list_of_thickness[i]*z0i*T0
        if i in index:
            N_T += Q_theta*alpha_theta*integral_DeltaT
            M_T += Q_theta*alpha_theta*integral_DeltaT_z
        else:
            pass

    return (N_T, M_T)

def rotated_lamina_stiffness_inplane(E1: float,
                                     E2: float,
                                     G12: float,
                                     nu12: float,
                                     theta: float) -> Form:
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