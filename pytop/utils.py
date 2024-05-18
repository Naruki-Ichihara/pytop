# 　_*_ coding: utf-8 _*_
'''Utilites for pytop, including: bridging between fenics variables to numpy array, or vice versa.'''

from fenics import *
from fenics_adjoint import *
import numpy as np
from typing import Callable, Iterable
from dataclasses import dataclass

def make_noiman_boundary_domains(mesh: Mesh, subdomains: Iterable[SubDomain], initialize=False) -> Measure:
    '''Create a Measure object for the boundary domains.

    Args: (Mesh, Iterable[SubDomain], bool)
        mesh: fenics mesh.
        subdomains: list of subdomains.
        initialize: whether to initialize the boundary markers.

    Returns: (Measure)
        Measure object for the boundary domains.
    '''
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    if initialize:
        boundary_markers.set_all(0)
    for i, subdomain in enumerate(subdomains):
        subdomain.mark(boundary_markers, i)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
    return ds

def fenics_function_to_np_array(fenics_variable: Constant
                                               | Function
                                               | GenericVector ) -> np.ndarray:
    '''Convert fenics variable to numpy array.

    Args: (Constant | Function | GenericVector)
        fenics_variable: fenics values to be converted.

    Raises:
        TypeError: if the input is not a fenics vector.

    Returns: (np.ndarray)
        numpy array.
    '''
    if isinstance(fenics_variable, Constant):
        return np.array(fenics_variable.values())

    elif isinstance(fenics_variable, Function):
        fenics_vector = fenics_variable.vector()
        if fenics_vector.mpi_comm().size > 1:
            gathered_fenics_vector = fenics_vector.gather(
                np.arange(fenics_vector.size(), dtype='I'))
        else:
            gathered_fenics_vector = fenics_vector.get_local()
        return np.asarray(gathered_fenics_vector)

    elif isinstance(fenics_variable, GenericVector):
        if fenics_variable.mpi_comm().size > 1:
            gathered_fenics_vector = fenics_variable.gather(
                np.arange(fenics_variable.size(), dtype='I'))
        else:
            gathered_fenics_vector = fenics_variable.get_local()
        return np.asarray(gathered_fenics_vector)

    else:
        raise TypeError(
            'Input is not a supported type. Supported types are: Constant, Function, GenericVector on fenics.')


def np_array_to_fenics_function(np_array: np.ndarray,
                                fenics_function: Function) -> Function:
    '''Convert numpy array to fenics variable.

    Args: (np.ndarray, Function)

        np_array: numpy array to be converted.
        fenics_function: fenics function to be assigned.

    Raises:
        TypeError: if the input is not a numpy array.
        ValueError: if the input numpy array is not of the same size as the fenics vector.

    Returns: (Function)
        fenics variable.

    '''
    if isinstance(fenics_function, Function):
        function_space = fenics_function.function_space()
        function = type(fenics_function)(function_space)
        fenics_vector_size = function.vector().size()
        np_array_size = np_array.size
        if np_array_size != fenics_vector_size:
            err_msg = (
                f"Cannot convert numpy array to Function: Wrong size {np_array_size} vs {fenics_vector_size}")
            raise ValueError(err_msg)

        if np_array.dtype != np.float_:
            err_msg = (
                f"The numpy array must be of type {np.float_}, but got {np_array.dtype}")
            raise ValueError(err_msg)

        range_begin, range_end = function.vector().local_range()
        localArray = np.asarray(np_array).reshape(
            fenics_vector_size)[range_begin:range_end]
        function.vector().set_local(localArray)
        function.vector().apply("insert")
        return function
    else:
        raise TypeError(
            'Input fenics vriable is not a supported type. Supported types is Function on fenics.')


def set_fields_to_fenics_function(fields: list[Callable[[Iterable], float]],
                                  function: Function) -> None:
    '''Set values for a fenics function.
    Elements of ```fields``` are assumed to be the followig pyfunction:
    ```python
    value1 = lambda x: f(x[0], x[1], ..., x[n]) # n is the dimension of the Function space.
    value2 = lambda x: g(x[0], x[1], ..., x[n]) 
    set_fields_to_fenics_function([value1, value2], function_space) # The rank of the function_space and length of the values must be the same.
    ```

    if the element is not a function but a constant value, it is assumed to be a constant value.

    ```python
    set_fields_to_fenics_function([1.0, 1.0], function_space)
    ```

    Args: (list, Function)
        values: list of values to be assigned.
        function: fenics function to be assigned.

    Raises:
        TypeError: if the input is not a list.

    '''
    if not isinstance(fields, list):
        raise TypeError('Input values must be a list.')

    class Field(UserExpression):
        def eval(self, value, x):
            for i, field in enumerate(fields):
                if not callable(field):
                    value[i] = field
                else:
                    value[i] = field(x)

        def value_shape(self):
            if len(fields) == 1:
                return ()
            else:
                return (len(fields),)

    function.interpolate(Field())
    return


def create_initialized_fenics_function(fields: list[Callable[[Iterable], float]],
                                       function_space: FunctionSpace) -> Function:
    '''Return a fenics function defined on the ```functionspace``` with values assigned.
    Elements of ```fields``` are assumed to be the following pyfunction:
    ```python
    value1 = lambda x: f(x[0], x[1], ..., x[n]) # n is the dimension of the Function space.
    value2 = lambda x: g(x[0], x[1], ..., x[n]) 
    create_initialized_fenics_function([value1, value2], functionspace) # The rank of the functionspace and length of the values must be the same.
    ```

    if the element is not a function but a constant value, it is assumed to be a constant value.

    ```python
    create_initialized_fenics_function([1.0, 1.0], functionspace)
    ```

    Args: (list, Function)
        values: list of values to be assigned.
        functionspace: fenics function space.

    Raises:
        TypeError: if the input is not a list.

    Returns: (Function)
        fenics function.

    '''
    if not isinstance(fields, list):
        raise TypeError('Input values must be a list.')
    function = Function(function_space)

    class Field(UserExpression):
        def eval(self, value, x):
            for i, field in enumerate(fields):
                if not callable(field):
                    value[i] = field
                else:
                    value[i] = field(x)

        def value_shape(self):
            if len(fields) == 1:
                return ()
            else:
                return (len(fields),)

    function.interpolate(Field())
    return function

@dataclass
class MPI_Communicator:
    '''MPI communicator for parallel computing.
    
    Attributes:
        comm_world (MPI_Comm): MPI communicator.
        rank (int): Rank of the process.
        size (int): Number of processes.
    '''
    comm_world = MPI.comm_world
    rank = MPI.comm_world.rank
    size = MPI.comm_world.size