# -*- coding: utf-8 -*-
''' Problem statement class.

'''

from abc import ABCMeta, abstractmethod
from fenics import *
from fenics_adjoint import *
import numpy as np
from pytop.utils import fenics_function_to_np_array
from pytop.designvariable import DesignVariables


class ProblemStatement(metaclass=ABCMeta):
    ''' Base class for problem statements.'''

    def __init__(self):
        self.index = 0
        pass

    @abstractmethod
    def objective(self, design_variables: DesignVariables, **kwargs) -> AdjFloat:
        raise NotImplementedError()

    def compute_sensitivities(self, 
                              design_variables: DesignVariables, 
                              target: str, 
                              variable_key: str) -> np.ndarray:

        control = Control(design_variables.dict_of_original_functions[variable_key])

        if not hasattr(self, target):
            raise AttributeError(f'The "{target}" is not defined.')
        try :
            compute_gradient(getattr(self, target)(design_variables), control)
        except AttributeError:
            # warnings.warn(f'The "{target}" is independent of the variable "{variable_key}".')
            return np.zeros(design_variables[variable_key].vector().size())
        sensitivity = fenics_function_to_np_array(compute_gradient(getattr(self, target)(design_variables),
                                       control))
        return sensitivity