"""Built-in solver for topology optimization problems. #TODO!!"""

from scipy.sparse import diags
from scipy.linalg import solve
import numpy as np

def solve_subproblem_primal_dual_interior_point_method():
    raise NotImplementedError()

def solve_gcmma_subproblem(current_iteration: int,
                         numbers_of_constraints: int,
                         current_variables: np.ndarray,
                         lower_bounds: np.ndarray,
                         upper_bounds: np.ndarray,
                         previous_variables: np.ndarray,
                         two_iters_ago_variables: np.ndarray,
                         current_evaluations: np.ndarray,
                         derivatives: np.ndarray,
                         previous_lower_asymptotes: np.ndarray,
                         previous_upper_asymptotes: np.ndarray):
    raise NotImplementedError()

def solve_mma_subproblem(current_iteration: int,
                         numbers_of_constraints: int,
                         current_variables: np.ndarray,
                         lower_bounds: np.ndarray,
                         upper_bounds: np.ndarray,
                         previous_variables: np.ndarray,
                         two_iters_ago_variables: np.ndarray,
                         current_evaluations: np.ndarray,
                         derivatives: np.ndarray,
                         previous_lower_asymptotes: np.ndarray,
                         previous_upper_asymptotes: np.ndarray):
    
    numbers_of_variables = len(current_variables)
    
    # Define moving asymptotes
    if current_iteration <= 2:
        lower_asymptotes = current_variables - 0.5 * (upper_bounds - lower_bounds)
        upper_asymptotes = current_variables + 0.5 * (upper_bounds - lower_bounds)

    else:
        evaluation = (current_variables - previous_variables) * (previous_variables - two_iters_ago_variables)

        if evaluation < 0:
            gamma = 0.7
        elif evaluation > 0:
            gamma = 1.2
        elif evaluation == 0:
            gamma = 1.0
        
        lower_asymptotes = current_variables - gamma * (previous_variables - previous_lower_asymptotes)
        upper_asymptotes = current_variables + gamma * (previous_upper_asymptotes - previous_variables)
    
    # Check satisfaction of bound constraints
    lower_minimum = current_variables - 10*(upper_bounds - lower_bounds)
    lower_maximum = current_variables - 0.01*(upper_bounds - lower_bounds)
    upper_minimum = current_variables + 0.01*(upper_bounds - lower_bounds)
    upper_maximum = current_variables + 10*(upper_bounds - lower_bounds)
    lower_asymptotes = np.maximum(lower_asymptotes, lower_minimum)
    lower_asymptotes = np.minimum(lower_asymptotes, lower_maximum)
    upper_asymptotes = np.maximum(upper_asymptotes, upper_minimum)
    upper_asymptotes = np.minimum(upper_asymptotes, upper_maximum)

    # Calcuilate the bounds, alpha and beta
    alpha = np.maximum(lower_bounds, lower_asymptotes+0.1*(current_variables-lower_asymptotes), current_variables-0.5*(upper_bounds-lower_bounds))
    beta = np.minimum(upper_bounds, upper_asymptotes-0.1*(upper_asymptotes-current_variables), current_variables+0.5*(upper_bounds-lower_bounds))