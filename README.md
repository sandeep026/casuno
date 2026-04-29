# **casuno**
casuno is a lightweight Python wrapper designed to bridge the gap between CasADi's intuitive Opti modeling interface and the Uno modular NLP solver.

Modeling optimization problems often involves tedious index bookkeeping and manual derivative setup. casuno lets you model in the user-friendly Opti stack and solve using the advanced, filter-based SQP and interior-point methods of Uno, all while CasADi handles the heavy lifting of Automatic Differentiation (AD) behind the scenes.

**Status**: Early-stage prototype (as of April 2026). This is experimental code and has not been thoroughly stress-tested for production environments.

(uno interface for casadi is still in developement as of 19-4-26)

### Key Features
- Hassle-free Modeling: Use `opti.variable()`, `opti.subject_to()`, and `opti.minimize()` without worrying about vector indices.
- Automatic Derivatives: Leverages CasADi's high-performance AD to provide gradients and Jacobians to Uno.
- Modular Solving: Easily switch between Uno's presets (like filtersqp) and configurations.
- unopy Integration: Seamlessly transforms CasADi problem structures into unopy models.

### Workflow
- Model your NLP using CasADi's Opti interface.
- Convert the problem into a unopy compatible model via `opti2unomodel()`.
- Configure your Uno solver (presets, Hessian models, etc.).
- Solve and extract results as standard NumPy arrays.

### Requirements

- python ^3.11
- casadi
- unopy

### Quick Start

Examples are contained in casuno.py. Race car problem from casadi's repository and move block example from Mathew Kelly's
collocation paper are solved. 

### Limitation

While functional, casuno is still in active development. Please keep the following in mind:
- Simple Bounds: Opti does not treat simple variable bounds ($\underline{x} \le x \le \bar{x}$) differently from general constraints. Consequently, casuno implements these as nonlinear constraints, which is less efficient than native solver bounds.
- Overhead: For extremely large-scale problems, the data conversion layer (CasADi DM ➔ NumPy ➔ Flattened C-style) may introduce noticeable overhead.
- Hessian Vector Products: The HVP operator is implemented but requires further verification for edge cases.

### Environment Note
This code was developed and tested on Windows. Remove `chp 65001` from `casuno.py` if it causes errors on your system.

### Quick start

The repository includes:

- Race Car Problem: A classic trajectory optimization benchmark from the CasADi repository.
- Move Block: Implementation of the collocation example from Matthew Kelly’s "How to do your own direct collocation"
- hs015: Hock-Schittkowski suite

#### code

```python
opti,x0=racecar()
#generate unopy model of NLP from opti
model=opti2unomodel(opti=opti,x0=x0)
# create solver class and configure
solver = unopy.UnoSolver()
solver.set_preset('filtersqp')
#by default bfgs
solver.set_option("hessian_model", "exact")
#solve the problem
result = solver.optimize(model)
print("\nOptimization Status:", result.optimization_status)
print("Solution Status:", result.solution_status)
print("Objective Value:", result.solution_objective)
print("cpu:", result.cpu_time)
print('iterations',result.number_iterations)
print('x',result.primal_solution)
```

#### Output

```bash
Active code page: 65001
Original model Python model
3003 variables, 4005 constraints (2003 equality, 2002 inequality)
Problem type: NLP
──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Radius    Phase ||Step||  Objective  Infeas    Statio    Compl     Status        
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 0      -      -         1.00e+01  OPT   -         1.000e+00  4.00e+00  1.00e+00  0.00e+00  initial point
 1      1      -         1.00e+01  OPT   -         -          -         -         -         infeasible    
 1      1      0.00e+00  1.00e+01  FEAS  1.47e+00  1.739e+00  2.41e+00  2.90e+00  1.97e-02  ✔ (Armijo)    
 2      1      0.00e+00  1.00e+01  FEAS  1.74e+00  0.000e+00  1.98e+00  1.59e+00  1.66e-02  ✔ (Armijo)    
 3      1      0.00e+00  1.00e+01  FEAS  1.68e+00  1.677e+00  7.98e-01  1.27e-01  6.49e-03  ✔ (Armijo)    
 4      1      0.00e+00  1.00e+01  FEAS  1.00e+00  2.085e+00  2.92e-01  4.64e+00  5.31e-04  ✔ (Armijo)    
 5      1      0.00e+00  1.00e+01  FEAS  1.03e+00  3.112e+00  4.48e-01  -         -         ✘ (Armijo)    
 -      2      -         5.13e-01  OPT   5.13e-01  2.063e+00  5.39e-03  1.06e+00  1.71e-04  ✔ (h-type)    
 6      1      -         1.03e+00  OPT   9.05e-01  1.903e+00  2.66e-02  9.97e-01  9.14e-04  ✔ (f-type)    
 7      1      -         1.03e+00  OPT   9.05e-01  1.905e+00  5.32e-05  1.02e-01  7.65e-07  ✔ (h-type)    
 8      1      -         1.03e+00  OPT   3.16e-01  1.905e+00  2.07e-07  4.69e-07  2.55e-09  ✔ (f-type)    
──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Radius    Phase ||Step||  Objective  Infeas    Statio    Compl     Status
──────────────────────────────────────────────────────────────────────────────────────────────────────────

Uno 2.7.1 (TR Fletcher-filter restoration inequality-constrained SQP method with exact Hessian and no inertia correction)
Sun Apr 19 02:53:46 2026
────────────────────────────────────────
Optimization status:                    Success
Solution status:                        Feasible KKT point
Objective value:                        1.904681
Primal feasibility:                     2.068204e-07
┌ Stationarity residual:                4.688917e-07
│ Primal feasibility:                   2.068204e-07
└ Complementarity residual:             2.551805e-09
CPU time:                               5.823s
Iterations:                             8
Objective evaluations:                  10
Constraints evaluations:                10
Objective gradient evaluations:         10
Jacobian evaluations:                   10
Hessian evaluations:                    10
Number of subproblems solved:           10

Optimization Status: OptimizationStatus.SUCCESS
Solution Status: SolutionStatus.FEASIBLE_KKT_POINT
Objective Value: 1.9046809064531998
cpu: 5.823
iterations 8
x [0.00000000e+00 0.00000000e+00 1.81275354e-06 ... 1.00000000e+00
 1.00000000e+00 1.90468091e+00]
```
