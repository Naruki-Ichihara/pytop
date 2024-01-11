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
    def __init__(self):
        self.index = 0
        pass

    @abstractmethod
    def objective(self, cotrols: list, **kwargs) -> AdjFloat:
        raise NotImplementedError()
    
    def computeSensitivities(self, targetResponce: str, targetControls: list) -> np.ndarray:
        ''' Compute sensitivities of a responce with respect to a target control.
        '''
        targetControls = [Control(targetControl) for targetControl in targetControls]
        sensitivities = fenics2np(compute_gradient(getattr(self, targetResponce)(targetControls), targetControls))
        return sensitivities