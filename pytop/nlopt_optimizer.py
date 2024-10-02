from fenics import *
from fenics_adjoint import *
import numpy as np
import pandas as pd
from typing import Optional
try:
    import nlopt as nlp
except ImportError:
    raise ImportError('nlopt is not installed.')
from pytop.designvariable import DesignVariables
from pytop.statement import ProblemStatement

class NloptOptimizer(nlp.opt):
    """This class is a wrapper for the nlopt optimization library.
    First, you need to construct this class with the design variables and the problem statement.
    In the first construct, the optimization algorithm must be defined. 
    The available algorithms are:

    - LD_MMA
    - LD_CCSAQ

    The optimization will lunch with calling ```run``` method.
    The options for the optimization algorithm can be set using the methods from the nlopt library.
    See, https://nlopt.readthedocs.io/en/latest/
    """
    def __init__(self,
                 design_variable: DesignVariables,
                 problem_statement: ProblemStatement,
                 algorithm: str = 'LD_MMA',
                 *args):
        super().__init__(getattr(nlp, algorithm), len(design_variable), *args)
        self.problem = problem_statement
        self.design_vector = design_variable
        self.__logging_dict = dict()

    def run(self, logging_path: Optional[str] = None):
        """Run the optimization. If ```logging_path``` is not None, the optimization history will be saved in a csv file.
        
        Args:
            logging_path (Optional[str], optional): Path to save the log. Defaults to None.
            
        """
        self.__logging_dict["objective"] = list()
        def eval(x, grad):
            self.design_vector.vector = x
            iter_num = self.design_vector.current_iteration_number
            cost = self.problem.objective(self.design_vector, iter_num)
            self.__logging_dict["objective"].append(cost)
            grads = [self.problem.compute_sensitivities(self.design_vector, "objective", key)
                     for key in self.design_vector.keys()]
            grad[:] = np.concatenate(grads)
            self.design_vector.update_counter()
            return cost
        
        def generate_cost_function(attribute, problem):
            def cost_function(x, grad):
                self.design_vector.vector = x
                iter_num = self.design_vector.current_iteration_number
                cost = getattr(problem, attribute)(self.design_vector, iter_num)
                self.__logging_dict[attribute] = cost
                grads = [problem.compute_sensitivities(self.design_vector, attribute, key)
                         for key in self.design_vector.keys()]
                grad[:] = np.concatenate(grads)
                return cost
            return cost_function
        
        for attribute in dir(self.problem):
            if attribute.startswith('constraint_'):
                cost_function = generate_cost_function(attribute, self.problem)
                self.add_inequality_constraint(cost_function, 1e-8)

        self.set_min_objective(eval)
        self.set_lower_bounds(self.design_vector.min_vector)
        self.set_upper_bounds(self.design_vector.max_vector)
        self.set_param("verbosity", 1)
        print(self.design_vector)
        result = self.optimize(self.design_vector.vector)

        if logging_path is not None:
            df = pd.DataFrame(self.__logging_dict)
            df.to_csv(logging_path)
        return result