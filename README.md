# **casuno**
A wrapper to solve optimization problems modelled in casadi's opti in modular NLP solver Uno.
As optimization problem is modeled in opti, user does not have to worry about book-keeping
decision variable indices, computation of derivatives. 

(uno interface for casadi is still in developement as of 19-4-26)


1. model optimization problem in casadi's opti
2. extract derivatives hassle free using AD
3. pass NLP information to unopy model
4. solve using any of Uno's solver

Examples are contained in casuno.py. Race car problem from casadi's repository and move block example from Mathew Kelly's
collocation paper are solved. 

Limitation
1. hessian operator needs more testing

**This is an early stage prototype code and has not been throughly tested.**
Happy to make changes to improve reliability of the code.

## Code
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

## Output

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
