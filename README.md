<h1 align="center"><b>pytop</b></h1>

## Significance

*pytop* is a library for the general purpose optimization in finite element space, including topology optimization. *pytop* provides straightforward coding for complex optimization problems.
*pytop* uses the [FEniCS](https://fenicsproject.org/) for finite element solver and [NLopt](https://github.com/stevengj/nlopt) as a optimization solver.

> *pytop* is the indipendent module of fenics and nlopt project.

> This software is based on Lagacy FEniCS (FEniCS2019.1.0). The new version, FEniCSx, is not supported.

## Introduction

The topology optimization is a common method to design physic-objective-oriented structures. *pytop* enables straightforward pythonic coding for high performanece
topology optimization. *pytop* may work with any general objective, physics, and (inequalilty) constraints, with automatic derivative.
*pytop* computes automatically any sensitivities with [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint) backend, so analytical sensitivity analysis is not needed.

Followings are major features of *pytop*, to name a few.

- **Multiphysics**: You can define objective and constraints of any physics through fenics infrastrctures.
- **Reuseable, simple coding**:
- **Scallable**: *pytop* supports message passing interface parallelaization.
- **Differentable**: The sensitivity is automatically computed by the pyadjoint backend.

## Install and getting ready

## Core modules

### DesignVariables

### ProblemStatement

### NloptOptimizer

## Citation
