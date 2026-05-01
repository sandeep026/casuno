import platform
import subprocess
from time import perf_counter as pf
from typing import Any

import casadi as cs
import numpy as np
import unopy

if platform.system() == "Windows":
    subprocess.run("chcp 65001", shell=True)


def opti2unomodel(opti: cs.Opti, x0: list) -> Any:
    """
    Solves a CasADi Opti problem using the Uno solver Python interface.

    all constraints assumed to be nonlinear (no simple bounds in opti)
    bound vectors are constant

    min. f
    s.t  lbg<=g<=ubg
    """
    t1 = pf()
    # extract decision vector objective constraint metadata
    x = opti.x
    f = opti.f
    g = opti.g
    nx = x.shape[0]
    ng = opti.ng
    lam_g = opti.lam_g

    # constants
    lbg = cs.evalf(opti.lbg).full().flatten().tolist()
    ubg = cs.evalf(opti.ubg).full().flatten().tolist()

    # casadi functions for numerical evaluation
    lam_f = cs.MX.sym("lam_f")
    lag = f * lam_f
    if ng:
        lag = lag - lam_g.T @ g
    # uno needs lower triangular part of hessian
    H, _ = cs.hessian(lag, x)
    H_lower = cs.tril(H)
    # extract nonzero entries using indices
    H_sp = H_lower.sparsity()
    hessian_row_indices, hessian_column_indices = H_sp.get_triplet()
    num_hessian_nonzeros = H_sp.nnz()
    H_nz = H_lower[H_sp]
    jit_op = {"jit": False, "jit_cleanup": True, "compiler": "shell"}
    H_func = cs.Function("H", [x, lam_f, lam_g], [H_nz], jit_op)
    # based on example for unopy types of arguments for these funcs. need clarity
    if ng == 0:
        mult_arr = cs.DM([])

    def lagrangian_hessian(x_arr, obj_multiplier, mult_arr, hess_vals):
        h_val = np.array(H_func(x_arr, obj_multiplier, mult_arr).nonzeros()).flatten()
        hess_vals[:] = h_val

    # hessian vector product for Kyrlov methods (faster)
    grad_L = cs.gradient(lag, x)
    v_sym = cs.MX.sym("v", opti.nx)
    h_v_prod = cs.jtimes(grad_L, x, v_sym)
    hv_func = cs.Function("hv", [x, lam_f, lam_g, v_sym], [h_v_prod], jit_op)

    def lagrangian_hessian_operator(x_arr, obj_multiplier, mult_arr, v_arr, hv_vals):
        res = hv_func(x_arr, obj_multiplier, mult_arr, v_arr).full().flatten()
        hv_vals[:] = res

    # casadi funcs. to unopy funcs.
    obj_func = cs.Function("obj", [x], [f], jit_op)
    grad_func = cs.Function("grad", [x], [cs.gradient(f, x)], jit_op)
    con_func = cs.Function("con", [x], [g], jit_op)
    jac_func = cs.Function("jac_f", [x], [cs.jacobian(g, x)], jit_op)

    # jacobian needs only nonzero entries
    # sniff it out using sparsity class
    jac_sparsity = jac_func.sparsity_out(0)
    num_jac_nonzeros = jac_sparsity.nnz()
    jac_row_indices, jac_col_indices = jac_sparsity.get_triplet()

    # unopy compatible functions
    def get_objective(x_arr):
        return float(obj_func(x_arr).full()[0, 0])  # float or ndarray?

    def get_gradient(x_arr, gradient):
        gradient[:] = grad_func(x_arr).full().flatten()

    def get_constraint(x_arr, constraint):
        constraint[:] = con_func(x_arr).full().flatten()

    # only nonzero entries

    def get_jacobian(x_arr, jacobian):
        jac_dm = jac_func(x_arr)
        jacobian[:] = np.array(jac_dm.nonzeros()).flatten()

    # cannot extract simple bounds on dec. vars. directly
    # additional steps required
    # assume unbounded but limit in nonlinear constraints
    model = unopy.Model(
        unopy.PROBLEM_NONLINEAR,  # api to specify type of optimization?
        nx,
        [-float("inf")] * nx,
        [float("inf")] * nx,
        unopy.ZERO_BASED_INDEXING,
    )

    model.set_objective(unopy.MINIMIZE, get_objective, get_gradient)
    if ng > 0:
        model.set_constraints(
            ng,
            get_constraint,
            lbg,
            ubg,
            num_jac_nonzeros,
            jac_row_indices,
            jac_col_indices,
            get_jacobian,
        )

    model.set_lagrangian_hessian(
        num_hessian_nonzeros,
        unopy.LOWER_TRIANGLE,
        hessian_row_indices,
        hessian_column_indices,
        lagrangian_hessian,
    )
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_NEGATIVE)
    # model.set_lagrangian_hessian_operator(lagrangian_hessian_operator)

    # guess for decision vector
    model.set_initial_primal_iterate(x0)
    tnet = pf() - t1
    print(f"opti2unopy construction time [s]: {tnet}")
    return model


def print_stats(result) -> None:
    print(f"Optimization Status: {result.optimization_status}")
    print(f"Solution Status    : {result.solution_status}")
    print(f"Objective Value    : {result.solution_objective}")
    print(f"cpu                : {result.cpu_time}")
    print(f"iterations         : {result.number_iterations}")


#    from cProfile import Profile
#    import pstats
#    with Profile() as p:
#    stats=pstats.Stats(p).sort_stats('cumtime')
#    stats.print_stats(50)
