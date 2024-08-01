# -*- coding: utf-8 -*-
''' Problem statement class.

'''

from abc import ABCMeta, abstractmethod
from fenics import *
from fenics_adjoint import *
import numpy as np
from pytop.utils import fenics_function_to_np_array
from pytop.designvariable import DesignVariables
from typing import Optional

def save(func):
    def wrapper(self, *args, **kwargs):
        print("Objective called")
        result = func(self, *args, **kwargs)
        print(type(result))
        return result
    return wrapper

class ProblemStatement(metaclass=ABCMeta):
    '''The ```ProblemStatement``` class is an abstract class for defining the optimization physics.
    The ```objective``` method must be implemented in the derived class. ```objective``` gives
    the ```DesignVariables``` as controls. You can access each design variable by its key like as:
    ```python
    x = design_variables["key"]
    ```
    The ```x``` is a ```Function``` object. The ```objective``` method must return a ```AdjFloat``` of
    the objective function value.
    ```python
    class Problem(ProblemStatement):
        def objective(self, design_variables):
            x = design_variables["key"]
            # do something with x
            solve(a == L, uh, bc)
            # Compute the objective function with the solution uh for example:
            return assemble(inner(f, uh) * dx)
    ```
    '''

    @abstractmethod
    def objective(self, design_variables: DesignVariables, iter_num: int | None, **kwargs) -> AdjFloat:
        '''The objective function. You must implement this method.

        Args:
            design_variables (DesignVariables): The design variables.
            iter_num (int): The iteration number. Iter_num has None when the optimizer calls the objective in backward mode.
            **kwargs: Optional arguments.

        Returns:
            AdjFloat: The objective function value.
        '''
        raise NotImplementedError()

    def compute_sensitivities(self, 
                              design_variables: DesignVariables, 
                              target: str, 
                              variable_key: str) -> np.ndarray:

        control = Control(design_variables.dict_of_original_functions[variable_key])

        if not hasattr(self, target):
            raise AttributeError(f'The "{target}" is not defined.')
        try :
            compute_gradient(getattr(self, target)(design_variables, None), control)
        except AttributeError:
            # warnings.warn(f'The "{target}" is independent of the variable "{variable_key}".')
            return np.zeros(design_variables[variable_key].vector().size())
        sensitivity = fenics_function_to_np_array(compute_gradient(getattr(self, target)(design_variables, None),
                                       control))
        return sensitivity
    
    def recorder(self, xdmf_file: XDMFFile, variable: Form | Function, function_space: FunctionSpace, name: str, iter_num: int | None):
        '''Record the variable to the xdmf file.

        Args:
            xdmf_file (XDMFFile): The xdmf file.
            variable (Form | Function): The variable to record.
            name (str): The name of the variable.
            iter_num (int): The iteration number.
        '''
        variable = project(variable, function_space)
        variable.rename(name, name)
        if iter_num is not None:
            xdmf_file.write(variable, iter_num)
        pass
