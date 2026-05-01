import unopy

from casuno import opti2unomodel, print_stats
from opti_problems import registry as models

# for discrete OCPs the discretization intervals can be set using n
# solve flag lets u solve nlp from within opti with IPOPT
opti, x0, xsol = models["kelly_ocp"](n=10000, solve=True)
# generate unopy model of NLP from opti
model = opti2unomodel(opti=opti, x0=x0)
solver = unopy.UnoSolver()
# create solver class and configure (ipopt/filtersqp)
# use the exact hessian. better than lbfgs is most cases tested
solver.set_preset("ipopt")
solver.set_option("hessian_model", "exact")
solver.set_option("inertia_correction_strategy", "primal")
result = solver.optimize(model)
print_stats(result=result)

opti, x0, xsol = models["racecar_ocp"](n=10000, solve=True)
model = opti2unomodel(opti=opti, x0=x0)
solver = unopy.UnoSolver()
solver.set_preset("ipopt")
solver.set_option("hessian_model", "exact")
solver.set_option("inertia_correction_strategy", "primal")
result = solver.optimize(model)
print_stats(result=result)


opti, x0, xsol = models["hs015_nlp"](solve=True)
model = opti2unomodel(opti=opti, x0=x0)
solver = unopy.UnoSolver()
solver.set_preset("ipopt")
solver.set_option("hessian_model", "exact")  # better than lbfgs
solver.set_option("inertia_correction_strategy", "primal")
result = solver.optimize(model)
print_stats(result=result)
