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
