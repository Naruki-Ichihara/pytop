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
        """Run the optimization.
        
        Args:
            logging_path (Optional[str], optional): Path to save the log. Defaults to None.
            
        """
        self.__logging_dict["objective"] = list()
        def eval(x, grad):
            self.design_vector.vector = x
            cost = self.problem.objective(self.design_vector)
            self.__logging_dict["objective"].append(cost)
            grads = [self.problem.compute_sensitivities(self.design_vector, "objective", key)
                     for key in self.design_vector.keys()]
            grad[:] = np.concatenate(grads)
            return cost
        
        def generate_cost_function(attribute, problem):
            def cost_function(x, grad):
                self.design_vector.vector = x
                cost = getattr(problem, attribute)(self.design_vector)
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
        result = self.optimize(self.design_vector.vector)

        if logging_path is not None:
            df = pd.DataFrame(self.__logging_dict)
            df.to_csv(logging_path)
        return result