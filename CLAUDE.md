# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pytop is a Python library for general-purpose optimization in finite element space (topology optimization) built on legacy FEniCS 2019.1.0 (not FEniCSx) and NLopt. It requires a Docker container with FEniCS pre-installed.

## Build & Development Commands

```bash
# Install package in development mode
pip install -e .

# Run all tests (must be inside Docker container)
docker-compose exec -T pytop pytest

# Run a single test
docker-compose exec -T pytop pytest tests/test_filter.py

# Generate API documentation
bash generate_doc.sh

# Start the Docker environment
docker-compose up -d
```

## Architecture

The optimization loop follows this pattern: **DesignVariables → ProblemStatement → NloptOptimizer**

- **`pytop/__init__.py`** — Re-exports all of FEniCS/fenics_adjoint plus pytop's public API. Users write `import pytop as pt` and get both FEniCS and pytop symbols.

- **`pytop/designvariable.py`** (`DesignVariables`) — Manages design variables (density fields, orientations, etc.) with bounds and pre/post-processing callbacks. Handles FEniCS Function ↔ NumPy array conversion for the optimizer.

- **`pytop/statement.py`** (`ProblemStatement`) — Abstract base class. Users implement `objective()` and any methods prefixed with `constraint_` which are auto-discovered as inequality constraints. Uses fenics-adjoint for automatic gradient computation via `compute_sensitivities()`.

- **`pytop/nlopt_optimizer.py`** (`NloptOptimizer`) — Wraps NLopt (LD_MMA or LD_CCSAQ algorithms). Auto-registers constraints from ProblemStatement, logs history to CSV.

- **`pytop/filters.py`** — Helmholtz filter (PDE-based density smoothing) and smoothed Heaviside filter (thresholding). Standard topology optimization filters.

- **`pytop/physics/`** — Reusable weak form definitions:
  - `elasticity.py` — 2D/3D isotropic and orthotropic linear elasticity
  - `shell.py` — Nagdi shell elements, multilayer composites
  - `thermal.py` — Poisson/thermal diffusion
  - `utils.py` — Penalization functions (SIMP/RAMP), stress computation

- **`pytop/toolkit/dehomogenization.py`** — Swift-Hohenberg equation solver for stripe pattern generation (fiber-reinforced composite design).

## Key Technical Details

- All automatic differentiation is handled by fenics-adjoint — do not manually compute gradients.
- MPI parallelism is supported via mpi4py; the `MPI_Communicator` dataclass in utils.py wraps MPI state.
- The Docker base image (`ichiharanaruki/pytop:latest`) includes FEniCS, NLopt, and compiled dependencies. Running outside Docker is not straightforward.
- NumPy is pinned to 1.26.4 for FEniCS compatibility.
