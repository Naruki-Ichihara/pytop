# -*- coding: utf-8 -*-
''' Problem statement class.

'''

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from fenics import *
from fenics_adjoint import *
import numpy as np
from .utils import np2fenics, fenics2np

class ProblemStatement(metaclass=ABCMeta):
    ''' Base class for problem statements.'''
    def __init__(self, initialIndex=0):
        self.initialIndex = initialIndex
        
    @abstractmethod
    def objective(self, cotrols: list, **kwargs) -> AdjFloat:
        raise NotImplementedError()
    
    def computeSensitivities(self, targetResponce: str, targetControl: list, **kwargs) -> np.ndarray:
        ''' Compute sensitivities of a responce with respect to a target control.
        '''
        return NotImplementedError() #TODO: implement this