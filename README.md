<h1 align="center"><b>pytop</b></h1>

Try out in codespace:

<a href='https://codespaces.new/Naruki-Ichihara/pytop'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

## Significance

*pytop* is a library for the general-purpose optimization in finite element space, including topology optimization. We provides straightforward coding for complex optimization problems.
*pytop* uses the [FEniCS](https://fenicsproject.org/) as finite element solver and [NLopt](https://github.com/stevengj/nlopt) as optimization solver.

> This software is the indipendent module of fenics and nlopt project.

> This software is based on Lagacy FEniCS (FEniCS2019.1.0). The new version, FEniCSx, is not supported.

> Only cpu based computation is supported. Now I consider developing the gpu based framework, please refer [gpytop](https://github.com/Naruki-Ichihara/gpytop).

## Documentation 🚧🚧Under construction🚧🚧
Documentation with many physics is available here:
<a href="https://naruki-ichihara.github.io/pytop_docs/"><strong>Documentation</strong></a>

## Introduction

Topology optimization is a common method for designing physic-objective-oriented structures. *pytop* enables straightforward pythonic coding for high performance
topology optimization. This software works with any general objective, physics, and (inequality) constraints, with automatic derivative.

## Install

We provide a container for this repository. The container includes python 3.11, FEniCS bundles, and NLOpt with python interface.
The container is avaiable in [dockerhub](https://hub.docker.com/repository/docker/ichiharanaruki/pytop/general).
To try out this repository, constract the codespace with following budge:

<a href='https://codespaces.new/Naruki-Ichihara/pytop'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces' style='max-width: 100%;'></a>

And install myself

```bash
pip instal .
```

## Core modules

### DesignVariables

### ProblemStatement

### NloptOptimizer

## Citation
