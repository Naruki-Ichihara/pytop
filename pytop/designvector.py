from fenics import *
from fenics_adjoint import *
import numpy as np
from pytop.utils import fenics2np, np2fenics

class DesignVectorVariables(dict):
    def __init__(self, variables: dict, **kwargs) -> None:
        """ Initialize design vector variables.

        Args:
            variables (dict): dictionary of variables.
        """
        super().__init__(**kwargs)
        self.fenicsLowerLimits = dict()
        self.__fnicsValues = dict()
        self.fenicsHigherLimits = dict()
        self.npLowerLimits = dict()
        self.__npValues = dict()
        self.npHigherLimits = dict()
        for key, functions in variables.items():
            self.fenicsLowerLimits[key] = functions[0]
            self.__fnicsValues[key] = functions[1]
            self.fenicsHigherLimits[key] = functions[2]
            self.npLowerLimits[key] = fenics2np(functions[0])
            self.__npValues[key] = fenics2np(functions[1])
            self.npHigherLimits[key] = fenics2np(functions[2])
            if not self.npLowerLimits[key].size == self.__npValues[key].size == self.npHigherLimits[key].size:
                raise ValueError(f'Size of domains of key "{key}" must be equal to the size of the initial values.')
            if np.all(self.npLowerLimits[key] > self.__npValues[key]):
                raise ValueError(f'Lower limits of key "{key}" must be smaller than the initial values.')
            if np.all(self.npHigherLimits[key] < self.__npValues[key]):
                raise ValueError(f'Higher limits of key "{key}" must be greater than the initial values.')
        self.__fieldCount = len(variables)
        self.__totalDesignVariableCount = sum([npValue.size for npValue in self.__npValues.values()])
        self.__npFullsizeVector = np.concatenate([npValue for npValue in self.__npValues.values()])
        pass

    def __len__(self) -> int:
        """ Return the number of fields.

        Returns: (int)
            number of fields.
        """
        return self.__fieldCount
    
    def __str__(self) -> str:
        string = f"Conut of fields: {self.__fieldCount}\nTotal count of design variables: {self.__totalDesignVariableCount}\nobjective ID: {id(self)}"
        return string
    
    @property
    def npFullsizeVector(self) -> np.ndarray:
        """ Return the full size design vector.

        Returns: (np.ndarray)
            full size design vector.
        """
        return self.__npFullsizeVector
    
    @npFullsizeVector.setter
    def npFullsizeVector(self, npFullsizeVector: np.ndarray) -> None:
        """ Set the full size design vector.

        Args:
            npFullsizeVector (np.ndarray): full size design vector.
        """
        if not npFullsizeVector.size == self.totalDesignVariableCount:
            raise ValueError('Size of npFullsizeVector must be equal to the number of design variables.')
        npFullsizeLowerLimits = np.concatenate([npLowerLimit for npLowerLimit in self.npLowerLimits.values()])
        if not np.all(npFullsizeVector >= npFullsizeLowerLimits):
            raise ValueError('npFullsizeVector must be greater than the lower limits.')
        npFullsizeHigherLimits = np.concatenate([npHigherLimit for npHigherLimit in self.npHigherLimits.values()])
        if not np.all(npFullsizeVector <= npFullsizeHigherLimits):
            raise ValueError('npFullsizeVector must be smaller than the higher limits.')
        splitIndices = []
        index = 0
        for npValue in self.npValues.values():
            index += npValue.size
            splitIndices.append(index)
        for key, npValue in zip(self.npValues.keys(), np.split(npFullsizeVector, splitIndices)):
            self.__npValues[key] = npValue
            self.__fnicsValues[key] = np2fenics(npValue, self.__fnicsValues[key])
        self.__npFullsizeVector = npFullsizeVector
        pass

    @property
    def fenicsValues(self) -> dict:
        """ Return the dictionary of fenics values.

        Returns: (dict)
            dictionary of fenics values.
        """
        return self.__fnicsValues
    
    @property
    def npValues(self) -> dict:
        """ Return the dictionary of numpy values.

        Returns: (dict)
            dictionary of numpy values.
        """
        return self.__npValues

    @property
    def fieldCount(self) -> int:
        """ Return the number of fields.

        Returns: (int)
            number of fields.
        """
        return self.__fieldCount
    
    @property
    def totalDesignVariableCount(self) -> int:
        """ Return the number of design variables.

        Returns: (int)
            number of design variables.
        """
        return self.__totalDesignVariableCount