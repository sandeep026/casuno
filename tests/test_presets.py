from itertools import product

import numpy.testing as npt
import pytest
import unopy

from casuno.casuno import opti2unomodel, print_stats
from casuno.opti_problems import registry as models

# for discrete OCPs the discretization intervals can be set using n
# solve flag lets u solve nlp from within opti with IPOPT

presets = ["ipopt", "filtersqp"]
cases = [
    (name, func, preset) for (name, func), preset in product(models.items(), presets)
]


@pytest.mark.parametrize("name,func,preset", cases)
def test_sol_ipopt_unopy(name, func, preset) -> None:
    opti, x0, xsol = func(solve=True)
    model = opti2unomodel(opti=opti, x0=x0)
    solver = unopy.UnoSolver()
    solver.set_preset(preset)
    solver.set_option("hessian_model", "exact")  # better than lbfgs
    solver.set_option("inertia_correction_strategy", "primal")
    result = solver.optimize(model)
    print_stats(result)
    print("-" * 80)
    print(f"testing - unopy solution vs. opti+IPOPT solution {name}")
    npt.assert_allclose(
        xsol.value(opti.f), result.solution_objective, rtol=1e-5, atol=1e-8
    )
