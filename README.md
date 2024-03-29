<h1 align="center"><b>pytop</b></h1>

Try out in codespace:

<a href='https://codespaces.new/Naruki-Ichihara/pytop'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

## Significance

*pytop* is a library for the general-purpose optimization in finite element space, including topology optimization. *pytop* provides straightforward coding for complex optimization problems.
*pytop* uses the [FEniCS](https://fenicsproject.org/) as finite element solver and [NLopt](https://github.com/stevengj/nlopt) as optimization solver.

> *pytop* is the indipendent module of fenics and nlopt project.

> This software is based on Lagacy FEniCS (FEniCS2019.1.0). The new version, FEniCSx, is not supported.

> Only cpu based computation is supported. Now I consider developing the gpu based framework, please refer [gpytop](https://github.com/Naruki-Ichihara/gpytop).

## Documentation
Documentation with many physics is available here:
<a href="https://naruki-ichihara.github.io/pytop_docs/"><strong>Documentation</strong></a>

> 🚧🚧Under construction🚧🚧
## Introduction

Topology optimization is a common method for designing physic-objective-oriented structures. *pytop* enables straightforward pythonic coding for high performance
topology optimization. *pytop* may work with any general objective, physics, and (inequality) constraints, with automatic derivative.
*pytop* computes automatically any sensitivities with the [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint) backend, so analytical sensitivity analysis is not needed.

Followings are major features of *pytop*, to name a few.

- **Multiphysics**
- **Reuseablity**
- **Scallablity**
- **Differentable**

## Install and getting ready

## Core modules

### DesignVariables

### ProblemStatement

### NloptOptimizer

## Citation
