from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
from typing import Callable, Iterable, Optional
import numpy as np
from pytop.utils import fenics_function_to_np_array, create_initialized_fenics_function


class DesignVariables():
    """Class for design variables.

    Attributes:
        __functions_dict (OrderedDict): Dictionary for the design variables.
        __controls_dict (OrderedDict): Dictionary for the control variables.
        __vector (np.ndarray): Design vector.
        __min_vector (np.ndarray): Minimum value of the design vector.
        __max_vector (np.ndarray): Maximum value of the design vector.
    
    """

    def __init__(self) -> None:
        self.__functions_dict = OrderedDict()
        self.__controls_dict = OrderedDict()
        self.__recording_dict = OrderedDict()
        self.__vector = np.array([])
        self.__min_vector = np.array([])
        self.__max_vector = np.array([])
        self.__counter = 0
        return
    
    def __len__(self) -> int:
        return len(self.__vector)
    
    def __str__(self) -> str:
        return "=================================================================\n" \
            f"Conut of fields: {len(self.__functions_dict)}\n" \
                 f"Total number of design variables: {len(self.__vector)}\n" \
                 f"object ID: {id(self)}\n" \
                 f"Keys of all design variables:\n{self.__functions_dict.keys()}\n" \
            "================================================================="

    def __getitem__(self, key: str) -> Function:
        return self.__functions_dict[key]
    
    def keys(self):
        return self.__functions_dict.keys()
    
    @property
    def dict_of_controls(self) -> OrderedDict:
        """Return the dictionary of control variables."""
        return self.__controls_dict
    
    @property
    def vector(self) -> np.ndarray:
        """Return the design vector."""
        return self.__vector
    
    @vector.setter
    def vector(self, value: np.ndarray) -> None:
        if not value.size == self.__vector.size:
            raise ValueError(f'Size mismatch. Expected size: {self.__vector.size}, but got: {value.size}')
        self.__vector = value
        split_index = []
        index = 0
        for function in self.__functions_dict.values():
            index += function.vector().size()
            split_index.append(index)
        
        # split the vector and assign to each function
        splited_vector = np.split(value, split_index)
        for function, vector in zip(self.__functions_dict.values(), splited_vector):
            function.vector()[:] = vector

        # Record the function
        for key, function in self.__recording_dict.items():
            function.write(self.__functions_dict[key], self.__counter)
        #self.__counter += 1
        return
    
    @property
    def min_vector(self) -> np.ndarray:
        """Return the minimum value of the design vector."""
        return self.__min_vector
    
    @min_vector.setter
    def min_vector(self, value: np.ndarray) -> None:
        raise NotImplementedError('This property is read-only.')
    
    @property
    def max_vector(self) -> np.ndarray:
        """Return the maximum value of the design vector."""
        return self.__max_vector
    
    @max_vector.setter
    def max_vector(self, value: np.ndarray) -> None:
        raise NotImplementedError('This property is read-only.')

    def register(self,
                 function_space: FunctionSpace,
                 function_name: str,
                 initial_value: list[Callable[[Iterable], float]],
                 range: tuple[float, float]
                      | tuple[Function, Function],
                 recording_path: Optional[str] = None) -> None:
        """Register a function to the design variables.
        
        Args:
        
            function_space (FunctionSpace): Function space for the design variavle.
            function_name (str): Name of the design variable.
            initial_value (list[Callable[[Iterable], float]]): Initial value of the function.
            range (tuple[float, float] | tuple[Function, Function]): Range of the function.
            recording (str): If you want to record the function, specify the file path.
            The function will be recorded in the ```{{path you provide}}/{{function name}}.xdmf```.
            
        Raises:
            ValueError: If the function name is already registered.
            
        """
        fenics_function = create_initialized_fenics_function(initial_value, function_space)
        fenics_function.rename(function_name, function_name)
        numpy_function = fenics_function_to_np_array(fenics_function)

        # register the function to dict
        if function_name in self.__functions_dict:
            raise ValueError(
                f'Function name "{function_name}" is already registered.')
        self.__functions_dict[function_name] = fenics_function
        self.__controls_dict[function_name] = Control(fenics_function)

        # register the function to vector
        self.__vector = np.append(self.__vector, numpy_function, axis=0)

        # register the range to min_vector and max_vector
        self.__min_vector = np.append(self.__min_vector,
                                      fenics_function_to_np_array(
                                      create_initialized_fenics_function([range[0]], function_space)))
        self.__max_vector = np.append(self.__max_vector,
                                      fenics_function_to_np_array(
                                      create_initialized_fenics_function([range[1]], function_space)))
        
        if recording_path is not None:
            self.__recording_dict[function_name] = XDMFFile(recording_path +"/"+ f'{function_name}.xdmf')
        return