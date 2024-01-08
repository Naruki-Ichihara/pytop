#　_*_ coding: utf-8 _*_
'''Utilites for pytop, including: bridging between fenics variables to numpy array, or vice versa.'''

from fenics import *
from fenics_adjoint import *
import numpy as np

def fenics2np(fenicsVar: Constant | Function | GenericVector) -> np.ndarray:
    '''Convert fenics variable to numpy array.
    
    Args: (Constant | Function | GenericVector)
        fenicsVar: fenics values to be converted.

    Raises:
        TypeError: if the input is not a fenics vector.
    
    Returns: (np.ndarray)
        numpy array.
    '''
    if isinstance(fenicsVar, Constant):
        return np.array(fenicsVar.values())
    
    elif isinstance(fenicsVar, Function):
        fenicsVector = fenicsVar.vector()
        if fenicsVector.mpi_comm().size > 1:
            gatheredFenicsVector = fenicsVector.gather(np.arange(fenicsVector.size(), dtyoe='I'))
        else:
            gatheredFenicsVector = fenicsVector.get_local()
        return np.asarray(gatheredFenicsVector)

    elif isinstance(fenicsVar, GenericVector):
        if fenicsVar.mpi_comm().size > 1:
            gatheredFenicsVector = fenicsVar.gather(np.arange(fenicsVar.size(), dtyoe='I'))
        else:
            gatheredFenicsVector = fenicsVar.get_local()
        return gatheredFenicsVector

    else:
        raise TypeError('Input is not a supported type. Supported types are: Constant, Function, GenericVector on fenics.')

def np2fenics(npArray: np.ndarray, fenicsFunction: Function) -> Function:
    '''Convert numpy array to fenics variable.
    
    Args: (np.ndarray, Function)

        npArray: numpy array to be converted.
        fenicsFunction: fenics function to be assigned.

    Raises:
        TypeError: if the input is not a numpy array.
        ValueError: if the input numpy array is not of the same size as the fenics vector.
    
    Returns: (Function)
        fenics variable.

    '''
    if isinstance(fenicsFunction, Function):
        functionSpace = fenicsFunction.function_space()
        u = type(fenicsFunction)(functionSpace)
        functionVectorSize = u.vector().size()
        npArraySize = npArray.size
        if npArraySize != functionVectorSize:
            err_msg = (f"Cannot convert numpy array to Function: Wrong size {npArraySize} vs {functionVectorSize}")
            raise ValueError(err_msg)

        if npArray.dtype != np.float_:
            err_msg = (f"The numpy array must be of type {np.float_}, but got {npArray.dtype}")
            raise ValueError(err_msg)

        rangeBegin, rangeEnd = u.vector().local_range()
        localArray = np.asarray(npArray).reshape(functionVectorSize)[rangeBegin:rangeEnd]
        u.vector().set_local(localArray)
        u.vector().apply("insert")
        return u
    else:
        raise TypeError('Input fenics vriable is not a supported type. Supported types is Function on fenics.')


    