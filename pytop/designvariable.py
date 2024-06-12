from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
from typing import Callable, Iterable, Optional
import numpy as np
from pytop.utils import fenics_function_to_np_array, create_initialized_fenics_function, MPI_Communicator


class DesignVariables():
    """The ```DesignVariables``` is a core class that manages the design variables.
    In the optimization problem in finite element analysis, followings are essential components:

    - The function space for the design variables.
    - The initial value of the design variables.
    - The range of the design variables.

    The ```register``` method can be used to define above components.
    The assigned functions can be accessed by standard dictionary-like syntax.

    ```python
    design_variables = DesignVariables()
    design_variables.register(function_space, "your_function_name", [initial], range)
    density = design_variables["your_function_name"]
    ```

    But we can not add some functions by the above syntax. Do not:

    ```python
    design_variables["new_function"] = new_function
    ```

    This code will be raise the read-only error. The ```register``` method should be used to add new functions.

    Attributes:
        keys (Iterable[str]): The names of the function.
        vector (np.ndarray): The flatten design vector.
        min_vector (np.ndarray): minimum range of the design vector.
        max_vector (np.ndarray): maximum range of the design vector.
        dict_of_original_functions (OrderedDict): The original functions.

    """
    def __init__(self) -> None:
        self.__functions_dict = OrderedDict()
        self.__vectors_dict = OrderedDict()
        self.__min_vector_dict = OrderedDict()
        self.__max_vector_dict = OrderedDict()
        self.__pre_process = OrderedDict()
        self.__post_process = OrderedDict()
        self.__recording_interval_dict = OrderedDict()
        self.__recording_dict = OrderedDict()
        self.__recording_dict_result = OrderedDict()
        self.__recording_dict_xml = OrderedDict()
        self.__counter = 0
        return
    
    def __len__(self) -> int:
        return len(self.vector)
    
    def __str__(self) -> str:
        return "=================================================================\n" \
            f"Conut of fields: {len(self.__functions_dict)}\n" \
                 f"Total number of design variables: {len(self.vector)}\n" \
                 f"object ID: {id(self)}\n" \
                 f"Keys of all design variables:\n{self.__functions_dict.keys()}\n" \
            "================================================================="

    def __getitem__(self, key: str) -> Function:
        return self.__pre_process[key](self.__functions_dict[key])
    
    def __contains__(self, key: str) -> bool:
        return key in self.__functions_dict
    
    def __iter__(self) -> Iterable[str]:
        return iter(self.__functions_dict)
    
    def keys(self) -> Iterable[str]:
        """Return the names of the functions."""
        return self.__functions_dict.keys()
    
    @property
    def current_iteration_number(self) -> int:
        """Return the current iteration number."""
        return self.__counter
    
    @current_iteration_number.setter
    def current_iteration_number(self, value: int) -> None:
        raise ValueError('This property is read-only.')
    
    def update_counter(self) -> None:
        """Update the counter."""
        self.__counter += 1
        return
    
    @property
    def dict_of_original_functions(self) -> OrderedDict:
        """Return the original functions that not applied any pre-process functions."""
        return self.__functions_dict
    
    @dict_of_original_functions.setter
    def dict_of_original_functions(self, key: str) -> None:
        raise ValueError('This property is read-only.')
    
    @property
    def vector(self) -> np.ndarray:
        """Return the flatten design vector."""
        flatten_vector = np.concatenate([vector for vector in self.__vectors_dict.values()], axis=0)
        return flatten_vector
    
    @vector.setter
    def vector(self, value: np.ndarray) -> None:
        if not value.size == self.vector.size:
            raise ValueError(f'Size mismatch. Expected size: {self.vector.size}, but got: {value.size}')
        split_index = []
        index = 0
        for function in self.__functions_dict.values():
            index += function.vector().size()
            split_index.append(index)
        
        # split the vector and assign to each function
        splited_vector = np.split(value, split_index)
        for function, vector in zip(self.__functions_dict.values(), splited_vector):
            range_begin, range_end = function.vector().local_range()
            insert_vector = vector[range_begin:range_end]
            function.vector().set_local(insert_vector)
            function.vector().apply("insert")

        # Record the function
        for key, function, result, result_xml in zip(self.__recording_dict.keys(), self.__recording_dict.values(), self.__recording_dict_result.values(), self.__recording_dict_xml.values()):
            if self.__counter%self.__recording_interval_dict[key] == 0:
                pre_processed_function = self.__post_process[key](self[key])
                pre_processed_function.rename(key, key)
                function.write(pre_processed_function, self.__counter)
                result.write(pre_processed_function)
                result_xml << pre_processed_function

        return
    
    @property
    def min_vector(self) -> np.ndarray:
        """Return the minimum value of the design vector."""
        flatten_min_vector = np.concatenate([vector for vector in self.__min_vector_dict.values()], axis=0)
        return flatten_min_vector
    
    @min_vector.setter
    def min_vector(self, value: np.ndarray) -> None:
        raise NotImplementedError('This property is read-only.')
    
    @property
    def max_vector(self) -> np.ndarray:
        """Return the maximum value of the design vector."""
        flatten_max_vector = np.concatenate([vector for vector in self.__max_vector_dict.values()], axis=0)
        return flatten_max_vector
    
    @max_vector.setter
    def max_vector(self, value: np.ndarray) -> None:
        raise NotImplementedError('This property is read-only.')

    def register(self,
                 function_space: FunctionSpace,
                 function_name: str,
                 initial_value: list[Callable[[Iterable], float]],
                 ranges: list[tuple[float, float]]
                      | list[tuple[Function, Function]],
                 pre_process: Optional[Callable[[Function], Function]] = None,
                 post_process: Optional[Callable[[Function], Function]] = None,
                 recording_path: Optional[str] = None,
                 recording_interval: int = 0,
                 mpi_comm: Optional[any] = None) -> None:
        """Register a function to the design variables.
        You should provide the followings:

        - The function space for the design variavle.
        - The name of the design variable. The name should be unique. 
          If the name is already registered, it will raise an error.
        - The initial value of the function.
        - The range of the function.
        - Some pre-process function. This function will be applied to the function in the optimization.
        - If you want to record the function, specify the file path.
          The function will be recorded in the ```{{path you provide}}/{{function name}}.xdmf```.

        The initial and range values can be a pyfunc as a following example:
            
        ```python
        initial_field = lambda x: pt.sin(x[0])+pt.cos(x[1])
        ```

        Note that the ```x``` is a list of spatial coordinates that contain the degree of freedom of the function space.
        All methods in the pyfunc should be implemented by the UFL functions.
        If the float value is provided, the function will be initialized by the constant value.

        The pre-process function can be used to apply some operations to the function.
        for example, the function can be filtered by the Helmholtz filter as follows:
        
        ```python
        filter = lambda x: pt.helmholtz_filter(x, R=0.05)
        ```
        
        Args:
            function_space (FunctionSpace): The function space for the design variable.
            function_name (str): The name of the function.
            initial_value (list[Callable[[Iterable], float]]): The initial value of the function.
            ranges (list[tuple[float, float]] | list[tuple[Function, Function]]): The range of the function.
            pre_process (Optional[Callable[[Function], Function]]): The pre-process function.
            post_process (Optional[Callable[[Function], Function]]): The post-process function. This function will be applied to the function in the recording. This does not affect the optimization.
            recording_path (Optional[str]): The path for the recording.
            recording_interval (int): The interval for the recording.
            mpi_comm (Optional[any]): The MPI communicator.
            
        Raises:
            ValueError: If the function name is already registered.
            
        """

        if pre_process is not None:
            self.__pre_process[function_name] = pre_process
        else:
            self.__pre_process[function_name] = lambda x: x
        
        if post_process is not None:
            self.__post_process[function_name] = post_process
        else:
            self.__post_process[function_name] = lambda x: x

        fenics_function = create_initialized_fenics_function(initial_value, function_space)
        fenics_function.rename(function_name, function_name)
        numpy_function = fenics_function_to_np_array(fenics_function)

        # register the function to dict
        if function_name in self.__functions_dict:
            raise ValueError(
                f'Function name "{function_name}" is already registered.')
        self.__functions_dict[function_name] = fenics_function

        self.__vectors_dict[function_name] = numpy_function
        range_mins = [range[0] for range in ranges]
        range_maxs = [range[1] for range in ranges]
        self.__min_vector_dict[function_name] = fenics_function_to_np_array(
            create_initialized_fenics_function(range_mins, function_space))
        self.__max_vector_dict[function_name] = fenics_function_to_np_array(
            create_initialized_fenics_function(range_maxs, function_space))

        
        if not np.all(self.min_vector <= self.vector):
            raise ValueError('Initial value is out of range.')
        if not np.all(self.max_vector >= self.vector):
            raise ValueError('Initial value is out of range.')
        
        if recording_path is not None:
            if mpi_comm is not None:
                self.__recording_dict[function_name] = XDMFFile(mpi_comm, recording_path +"/"+ f'{function_name}_history.xdmf')
                self.__recording_dict_result[function_name] = XDMFFile(mpi_comm, recording_path +"/"+ f'{function_name}_result.xdmf')
            else:
                self.__recording_dict[function_name] = XDMFFile(recording_path +"/"+ f'{function_name}_history.xdmf')
                self.__recording_dict_result[function_name] = XDMFFile(recording_path +"/"+ f'{function_name}_result.xdmf')
                self.__recording_dict_xml[function_name] = File(recording_path +"/"+ f'{function_name}.xml')
            self.__recording_interval_dict[function_name] = recording_interval
        return