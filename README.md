<p align="center">
   <img src="casunologo.png" alt="Uno" width="100%" />
</p>

## **CasUno**

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

Refer to the `.toml` file for details.

- python ^3.11
- casadi 3.7.2
- unopy 0.4.5

### Limitation

While functional, casuno is still in active development. Please keep the following in mind:

- Simple Bounds: Opti does not treat simple variable bounds ($\underline{x} \le x \le \bar{x}$) differently from general constraints. Consequently, casuno implements these as nonlinear constraints, which is less efficient than native solver bounds.
- Overhead: For extremely large-scale problems, the data conversion layer (CasADi DM ➔ NumPy ➔ Flattened C-style) may introduce noticeable overhead.
- Hessian Vector Products: The HVP operator is implemented but requires further verification for edge cases.

**Note**: _Not a limiation of the wrapper but it is important to construct the NLP using Opti() such that the expression graph is efficient. As this user
dependent, it adviced to follow best practices mentioned in casadi's Github Wiki_.

### Quick start

The repository includes:

- Race Car Problem: A classic trajectory optimization benchmark from the CasADi repository.
- Move Block: Implementation of the collocation example from Matthew Kelly’s "How to do your own direct collocation"
- hs015: Hock-Schittkowski suite

- `opti_problems.py` -  opti models are created here and stored in a dict registry
- `test.py` - solves the optimization within casadi's IPOPT and unopy's IPOPT present and checks
- `examples.py` - use this run all the problems (nlp an Ocps) from `opti_problem.py`

#### code

```python
from opti_problems import registry as models
from opti2unopy import opti2unomodel, print_stats
import unopy

# for discrete OCPs the discretization intervals can be set using n
# solve flag lets u solve nlp from within opti with IPOPT
opti,x0,xsol=models['kelly_ocp'](n=1000,solve=True)
#generate unopy model of NLP from opti 
model=opti2unomodel(opti=opti,x0=x0)   
solver = unopy.UnoSolver() 
# create solver class and configure (ipopt/filtersqp)     
# use the exact hessian. better than lbfgs is most cases tested     
solver.set_preset('ipopt')
solver.set_option("hessian_model", "exact")
solver.set_option('inertia_correction_strategy','primal')
result = solver.optimize(model)
print_stats(result=result)

opti,x0,xsol=models['racecar_ocp'](n=1000,solve=True) 
model=opti2unomodel(opti=opti,x0=x0)   
solver = unopy.UnoSolver()            
solver.set_preset('ipopt')
solver.set_option("hessian_model", "exact") 
solver.set_option('inertia_correction_strategy','primal')
result = solver.optimize(model)
print_stats(result=result)


opti,x0,xsol=models['hs015_nlp'](solve=True) 
model=opti2unomodel(opti=opti,x0=x0) 
solver = unopy.UnoSolver()           
solver.set_preset('ipopt')
solver.set_option("hessian_model", "exact") # better than lbfgs
solver.set_option('inertia_correction_strategy','primal')
result = solver.optimize(model)
print_stats(result=result)
```

#### Output

```bash******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

Total number of variables............................:     3003
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:     2004
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0


Number of Iterations....: 1

                                   (scaled)                 (unscaled)
Objective...............:   6.0000060000059996e+00    6.0000060000059996e+00
Dual infeasibility......:   1.7763568394002505e-15    1.7763568394002505e-15
Constraint violation....:   2.7040002181788481e-16    2.7040002181788481e-16
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   1.7763568394002505e-15    1.7763568394002505e-15


Number of objective function evaluations             = 2
Number of objective gradient evaluations             = 2
Number of equality constraint evaluations            = 2
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 2
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 1
Total seconds in IPOPT                               = 0.036

EXIT: Optimal Solution Found.
      solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
       nlp_f  |        0 (       0)  20.00us ( 10.00us)         2
       nlp_g  |  13.00ms (  6.50ms) 658.00us (329.00us)         2
  nlp_grad_f  |        0 (       0)  43.00us ( 14.33us)         3
  nlp_hess_l  |        0 (       0)  15.00us ( 15.00us)         1
   nlp_jac_g  |        0 (       0)   1.17ms (389.00us)         3
       total  |  46.00ms ( 46.00ms)  37.23ms ( 37.23ms)         1
opti construction+ IPOPT solve time [s]: 0.25543569999717874
opti2unopy construction time [s]: 0.14171770000029937
Original model Python model
3003 variables, 2004 constraints (2004 equality, 0 inequality)
Problem type: NLP
Reformulated model Python model -> no fixed bounds -> equality constrained -> bounds relaxed
3003 variables, 2004 constraints (2004 equality, 0 inequality)
The problem has no inequalities, picking a pure SQP method

Non-default options:
linear_solver = MUMPS

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Barrier   Steplength Phase Regulariz ||Step||  Objective  Infeas    Statio    Compl     Status        
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 0      -      -         -         -          OPT   -         -         5.005e-01  5.00e+00  1.00e-03  0.00e+00  initial point 
 1      1      -         -         1.00e+00   OPT   0.00e+00  6.99e+00  6.000e+00  3.00e-13  1.78e-14  0.00e+00  ✔ (h-type)    
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Barrier   Steplength Phase Regulariz ||Step||  Objective  Infeas    Statio    Compl     Status        
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Uno 2.7.1 (LS Waechter-filter restoration pure SQP method with exact Hessian and primal inertia correction)
Fri May  1 06:18:55 2026
────────────────────────────────────────
Optimization status:                    Success
Solution status:                        Feasible KKT point
Objective value:                        6.000006
Primal feasibility:                     2.996752e-13
┌ Stationarity residual:                1.776357e-14
│ Primal feasibility:                   2.996752e-13
└ Complementarity residual:             0
CPU time:                               0.032s
Iterations:                             1
Objective evaluations:                  2
Constraints evaluations:                2
Objective gradient evaluations:         2
Jacobian evaluations:                   2
Hessian evaluations:                    1
Number of subproblems solved:           1
Optimization Status: OptimizationStatus.SUCCESS
Solution Status    : SolutionStatus.FEASIBLE_KKT_POINT
Objective Value    : 6.000006000006482
cpu                : 0.032
iterations         : 1
Total number of variables............................:     3003
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:     2003
Total number of inequality constraints...............:     2002
        inequality constraints with only lower bounds:        1
   inequality constraints with lower and upper bounds:     1000
        inequality constraints with only upper bounds:     1001


Number of Iterations....: 29

                                   (scaled)                 (unscaled)
Objective...............:   1.9046834150970160e+00    1.9046834150970160e+00
Dual infeasibility......:   3.2030740282351644e-09    3.2030740282351644e-09
Constraint violation....:   9.2370555648813024e-14    9.2370555648813024e-14
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   4.1985839630552773e-09    4.1985839630552773e-09
Overall NLP error.......:   4.1985839630552773e-09    4.1985839630552773e-09


Number of objective function evaluations             = 31
Number of objective gradient evaluations             = 30
Number of equality constraint evaluations            = 31
Number of inequality constraint evaluations          = 31
Number of equality constraint Jacobian evaluations   = 30
Number of inequality constraint Jacobian evaluations = 30
Number of Lagrangian Hessian evaluations             = 29
Total seconds in IPOPT                               = 0.744

EXIT: Optimal Solution Found.
      solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
       nlp_f  |        0 (       0)  35.00us (  1.13us)        31
       nlp_g  |  17.00ms (548.39us)   9.40ms (303.29us)        31
  nlp_grad_f  |        0 (       0) 174.00us (  5.61us)        31
  nlp_hess_l  |  69.00ms (  2.38ms)  41.83ms (  1.44ms)        29
   nlp_jac_g  |  23.00ms (741.94us)  40.77ms (  1.32ms)        31
       total  | 749.00ms (749.00ms) 744.94ms (744.94ms)         1
opti construction+ IPOPT solve time [s]: 0.7915601000022434
opti2unopy construction time [s]: 0.029683599997952115
Original model Python model
3003 variables, 4005 constraints (2003 equality, 2002 inequality)
Problem type: NLP
Reformulated model Python model -> no fixed bounds -> equality constrained -> bounds relaxed
5005 variables, 4005 constraints (4005 equality, 0 inequality)

Non-default options:
linear_solver = MUMPS

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Barrier   Steplength Phase Regulariz ||Step||  Objective  Infeas    Statio    Compl     Status        
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 0      -      -         -         -          OPT   -         -         1.000e+00  2.30e+01  1.00e+00  9.90e-01  initial point 
 1      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  2.20e-02  1.002e+00  2.29e+01  2.20e+00  9.93e-01  ✔ (h-type)    
 2      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  2.29e-01  1.037e+00  2.30e+01  4.98e+00  1.01e+00  ✔ (h-type)    
 3      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  3.66e+00  1.664e+00  1.73e+03  8.28e+02  1.46e+00  ✔ (h-type)    
 4      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  2.74e+00  1.883e+00  9.56e+01  8.73e+01  1.62e-01  ✔ (h-type)    
 5      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  3.79e-01  1.918e+00  5.82e+01  5.30e+01  1.08e-01  ✔ (h-type)    
 6      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  3.46e-01  2.083e+00  3.67e+01  3.45e+01  9.94e-02  ✔ (h-type)    
 7      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  9.52e-01  2.440e+00  1.95e-01  5.38e+00  7.03e-02  ✔ (h-type)    
 8      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  4.30e-01  2.869e+00  1.21e-01  7.76e-01  3.11e-02  ✔ (h-type)    
 9      1      -         2.00e-02  1.00e+00   OPT   0.00e+00  5.53e-01  3.422e+00  1.09e+01  2.16e-01  2.33e-02  ✔ (h-type)    
 10     1      -         2.00e-02  1.00e+00   OPT   0.00e+00  1.09e-01  3.508e+00  1.84e+00  3.15e-02  2.25e-02  ✔ (h-type)    
 11     1      -         2.83e-03  1.00e+00   OPT   0.00e+00  1.92e-01  3.316e+00  6.73e-02  6.72e-02  4.15e-03  ✔ (h-type)    
 12     1      -         2.83e-03  1.00e+00   OPT   0.00e+00  5.22e-01  2.794e+00  6.24e-01  4.73e-02  3.34e-03  ✔ (h-type)    
 13     1      -         2.83e-03  1.00e+00   OPT   0.00e+00  6.75e-02  2.862e+00  3.39e-02  3.99e-04  2.84e-03  ✔ (h-type)    
 14     1      -         1.50e-04  1.00e+00   OPT   0.00e+00  3.85e-01  2.477e+00  3.12e-01  1.14e-02  9.15e-04  ✔ (h-type)    
 15     1      -         1.50e-04  1.00e+00   OPT   0.00e+00  3.60e-01  2.117e+00  5.86e-01  4.54e-02  3.02e-04  ✔ (h-type)    
 16     1      -         1.50e-04  1.00e+00   OPT   0.00e+00  1.66e-01  2.024e+00  6.37e-01  -         -         ✘ (h-type)    
 -      2      -         -         5.00e-01   -     -         8.29e-02  2.071e+00  4.52e-01  2.35e-02  2.28e-04  ✔ (h-type)    
 17     1      -         1.50e-04  1.00e+00   OPT   0.00e+00  7.17e-02  2.031e+00  8.81e-02  6.92e-04  1.55e-04  ✔ (h-type)    
 18     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  9.03e-02  1.961e+00  1.29e-01  1.43e-03  6.51e-05  ✔ (h-type)    
 19     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  2.18e-01  1.936e+00  8.76e-02  2.66e-02  3.68e-05  ✔ (h-type)    
 20     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  2.41e-01  1.917e+00  4.31e-02  5.05e-02  1.59e-05  ✔ (h-type)    
 21     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  3.72e-01  1.908e+00  1.07e-02  2.08e-02  9.39e-06  ✔ (h-type)    
 22     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  4.37e-01  1.907e+00  1.29e-04  1.93e-05  4.08e-06  ✔ (h-type)    
 23     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  1.88e-01  1.907e+00  5.69e-08  -         -         ✘ (f-type)    
 -      2      -         -         5.00e-01   -     -         9.42e-02  1.907e+00  6.46e-05  5.51e-02  2.60e-06  ✔ (f-type)    
 24     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  7.57e-02  1.907e+00  1.32e-08  1.99e-06  1.89e-06  ✔ (h-type)    
 25     1      -         2.51e-09  1.00e+00   OPT   0.00e+00  1.82e-01  1.905e+00  3.38e-05  1.13e-02  8.75e-07  ✔ (f-type)    
 26     1      -         2.51e-09  1.00e+00   OPT   0.00e+00  2.06e-01  1.905e+00  2.02e-05  1.84e-02  4.71e-07  ✔ (f-type)    
 27     1      -         2.51e-09  1.00e+00   OPT   0.00e+00  3.05e-01  1.905e+00  4.29e-06  5.67e-02  1.76e-07  ✔ (f-type)    
 28     1      -         2.51e-09  1.00e+00   OPT   0.00e+00  2.74e-02  1.905e+00  6.48e-08  3.78e-02  9.57e-08  ✔ (f-type)    
 29     1      -         2.51e-09  1.00e+00   OPT   0.00e+00  4.94e-03  1.905e+00  1.38e-11  7.55e-03  2.48e-08  ✔ (f-type)    
 30     1      -         2.51e-09  1.00e+00   OPT   0.00e+00  3.26e-03  1.905e+00  4.05e-13  2.97e-09  4.07e-09  ✔ (f-type)    
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Barrier   Steplength Phase Regulariz ||Step||  Objective  Infeas    Statio    Compl     Status        
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Uno 2.7.1 (LS Waechter-filter restoration primal-dual interior-point method with exact Hessian and primal inertia correction)
Fri May  1 06:18:57 2026
────────────────────────────────────────
Optimization status:                    Success
Solution status:                        Feasible KKT point
Objective value:                        1.904683
Primal feasibility:                     4.046774e-13
┌ Stationarity residual:                2.967755e-09
│ Primal feasibility:                   4.046774e-13
└ Complementarity residual:             4.067326e-09
CPU time:                               0.461s
Iterations:                             30
Objective evaluations:                  33
Constraints evaluations:                34
Objective gradient evaluations:         33
Jacobian evaluations:                   33
Hessian evaluations:                    30
Number of subproblems solved:           30
Optimization Status: OptimizationStatus.SUCCESS
Solution Status    : SolutionStatus.FEASIBLE_KKT_POINT
Objective Value    : 1.904683414990099
cpu                : 0.461
iterations         : 30
Total number of variables............................:        2
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:        3
        inequality constraints with only lower bounds:        2
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        1


Number of Iterations....: 16

                                   (scaled)                 (unscaled)
Objective...............:   1.2738984854970598e+01    3.0649997561059257e+02
Dual infeasibility......:   4.3165471197426086e-13    1.0385612370100716e-11
Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   2.5062671019034313e-09    6.0300786471796554e-08
Overall NLP error.......:   2.5062671019034313e-09    6.0300786471796554e-08


Number of objective function evaluations             = 25
Number of objective gradient evaluations             = 17
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 25
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 17
Number of Lagrangian Hessian evaluations             = 16
Total seconds in IPOPT                               = 0.009

EXIT: Optimal Solution Found.
      solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
       nlp_f  |        0 (       0)  29.00us (  1.16us)        25
       nlp_g  |        0 (       0)  31.00us (  1.24us)        25
  nlp_grad_f  |        0 (       0)  24.00us (  1.33us)        18
  nlp_hess_l  |        0 (       0)  49.00us (  3.06us)        16
   nlp_jac_g  |        0 (       0)  37.00us (  2.06us)        18
       total  |  10.00ms ( 10.00ms)  10.07ms ( 10.07ms)         1
opti construction+ IPOPT solve time [s]: 0.014597200002754107
opti2unopy construction time [s]: 0.0013611000031232834
Original model Python model
2 variables, 3 constraints (0 equality, 3 inequality)
Problem type: NLP
Reformulated model Python model -> no fixed bounds -> equality constrained -> bounds relaxed
5 variables, 3 constraints (3 equality, 0 inequality)

Non-default options:
linear_solver = MUMPS

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Barrier   Steplength Phase Regulariz ||Step||  Objective  Infeas    Statio    Compl     Status        
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 0      -      -         -         -          OPT   -         -         9.090e+02  4.02e+00  2.41e+03  2.50e+00  initial point 
 1      1      -         1.00e-01  1.00e+00   OPT   0.00e+00  9.90e-03  9.006e+02  4.00e+00  2.39e+03  7.72e-01  ✔ (h-type)    
 2      1      -         1.00e-01  1.00e+00   OPT   0.00e+00  7.04e-02  7.534e+02  3.85e+00  2.96e+03  6.82e-01  ✔ (h-type)    
 3      1      -         1.00e-01  1.00e+00   OPT   0.00e+00  1.95e+00  2.338e+01  1.24e+00  4.19e+03  1.26e-01  ✔ (h-type)    
 4      1      -         1.00e-01  1.00e+00   OPT   0.00e+00  5.47e-01  1.034e+01  1.54e+00  4.31e+03  8.15e-02  ✔ (h-type)    
 5      1      -         1.00e-01  1.00e+00   OPT   0.00e+00  2.30e+00  4.508e+02  5.68e+00  -         -         ✘ (h-type)    
 -      2      -         -         5.00e-01   -     -         1.15e+00  8.305e+01  2.19e+00  -         -         ✘ (h-type)    
 -      3      -         -         2.50e-01   -     -         5.75e-01  9.160e+00  1.51e+00  9.95e+03  1.26e-01  ✔ (h-type)    
 6      1      -         1.00e-01  1.00e+00   OPT   1.34e+04  2.88e-02  7.901e+00  1.47e+00  9.31e+03  5.53e-02  ✔ (h-type)    
 7      1      -         1.00e-01  1.00e+00   OPT   1.34e+04  2.88e-04  7.892e+00  1.47e+00  9.33e+03  5.54e-02  ✔ (h-type)    
 8      1      -         1.00e-01  1.00e+00   OPT   1.34e+04  6.23e-04  7.910e+00  1.47e+00  9.29e+03  5.53e-02  ✔ (h-type)    
 9      1      -         1.00e-01  1.00e+00   OPT   1.34e+04  5.18e-01  3.238e+01  1.24e+00  8.83e+04  7.03e+01  ✔ (h-type)    
 10     1      -         1.00e-01  1.00e+00   OPT   0.00e+00  1.99e-03  3.239e+01  1.24e+00  6.60e+04  4.55e+01  ✔ (h-type)    
 11     1      -         1.00e-01  1.00e+00   OPT   0.00e+00  2.20e-01  4.483e+01  1.14e+00  5.57e+04  3.92e+01  ✔ (h-type)    
 12     1      -         1.00e-01  1.00e+00   OPT   0.00e+00  2.59e+00  3.068e+02  1.17e+00  -         -         ✘ (h-type)    
 -      2      -         -         5.00e-01   -     -         1.29e+00  1.465e+02  8.65e-01  1.77e+04  1.04e+01  ✔ (h-type)    
 13     1      -         1.00e-01  1.00e+00   OPT   0.00e+00  2.17e+00  3.067e+02  2.93e-01  7.33e+03  4.00e+00  ✔ (h-type)    
 14     1      -         1.00e-01  1.00e+00   OPT   0.00e+00  2.93e-01  3.067e+02  1.28e-10  5.37e-02  2.27e-01  ✔ (h-type)    
 15     1      -         1.50e-04  1.00e+00   OPT   0.00e+00  2.73e-03  3.065e+02  4.74e-07  9.14e-05  2.15e-04  ✔ (f-type)    
 16     1      -         1.84e-06  1.00e+00   OPT   0.00e+00  3.03e-06  3.065e+02  8.83e-13  1.45e-10  1.85e-06  ✔ (f-type)    
 17     1      -         2.51e-09  1.00e+00   OPT   0.00e+00  3.68e-08  3.065e+02  9.99e-16  4.55e-13  2.51e-09  ✔ (f-type)    
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iterations
 Major  Minor  Penalty   Barrier   Steplength Phase Regulariz ||Step||  Objective  Infeas    Statio    Compl     Status        
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Uno 2.7.1 (LS Waechter-filter restoration primal-dual interior-point method with exact Hessian and primal inertia correction)
Fri May  1 06:18:57 2026
────────────────────────────────────────
Optimization status:                    Success
Solution status:                        Feasible KKT point
Objective value:                        306.5
Primal feasibility:                     9.992007e-16
┌ Stationarity residual:                4.547474e-13
│ Primal feasibility:                   9.992007e-16
└ Complementarity residual:             2.506011e-09
CPU time:                               0.088s
Iterations:                             17
Objective evaluations:                  21
Constraints evaluations:                22
Objective gradient evaluations:         21
Jacobian evaluations:                   21
Hessian evaluations:                    17
Number of subproblems solved:           17
Optimization Status: OptimizationStatus.SUCCESS
Solution Status    : SolutionStatus.FEASIBLE_KKT_POINT
Objective Value    : 306.4999754950125
cpu                : 0.088
iterations         : 17
```
