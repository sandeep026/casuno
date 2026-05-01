import numpy.testing as npt
import unopy

from opti2unopy import opti2unomodel, print_stats
from opti_problems import registry as models

# for discrete OCPs the discretization intervals can be set using n
# solve flag lets u solve nlp from within opti with IPOPT

for name, func in models.items():
    opti, x0, xsol = func(solve=True)
    model = opti2unomodel(opti=opti, x0=x0)
    solver = unopy.UnoSolver()
    solver.set_preset("ipopt")
    solver.set_option("hessian_model", "exact")  # better than lbfgs
    solver.set_option("inertia_correction_strategy", "primal")
    result = solver.optimize(model)
    print("#" * 50)
    print(f"testing - unopy solution vs. opti+IPOPT solution {name}")
    npt.assert_allclose(
        xsol.value(opti.f), result.solution_objective, rtol=1e-5, atol=1e-8
    )
