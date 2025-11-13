# mpils
MPILS: Multi-Phase Iterated Local Search Tuner

MPILS is an optimization algorithm for tuning MILP (Mixed-Integer Linear Programming) solvers such as ILOG Cplex or Gurobi.
These solvers get several parameters that the user can tune to efficiently improve the performance of the solver.
But tuning the solver is extremely difficult and the use of a tuning algorithm is very helpful.
MPILS is such a tuning algorithm based on the Iterated Local Search (ILS) metaheuristic with a multi-phase approach to efficiently explore the parameter space.
See the [paper](https://doi.org/10.1016/j.cor.2023.106344) for more details.

This repository contains my personal implementation of the MPILS algorithm.

## Doxygen Documentation
The project provides a Doxygen documentation that can be generated using the provided `Doxyfile`. To generate the documentation, run the following command in the terminal:
```doxygen Doxyfile```
This will create the documentation in the `docs` directory.
Only header files are documented for Doxygen. The source files are not documented but have some comments that should suffice.