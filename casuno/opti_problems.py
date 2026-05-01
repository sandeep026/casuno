from time import perf_counter as pf

import casadi as cs

registry = {}


def register(func):
    registry[func.__name__] = func
    return func


def get_initial(opti: cs.Opti) -> list:
    """
    obtain the initial value for the decision vector as a python list

    Parameters
    ----------
    opti : cs.Opti
        opti object

    Returns
    -------
    list
        list of initial values for decision vector
    """
    return opti.debug.value(opti.x, opti.initial()).flatten().tolist()


# naming function_name_ocp or function_name_nlp
# if nlp no additional argument
# if ocp, pass number of discretization intervals as argument
# solve - create optimization problem in opti and solve with IPOPT
# test - return the nlp solution from ipopt
@register
def kelly_ocp(n=200, solve=False) -> tuple:
    """
    kelly block problem

    direct transcription -> QP

    Optimal objective : 6

    Returns
    -------
    tuple
        (opti,initial guess)
    """
    t1 = pf()
    N = n  # number of control intervals
    opti = cs.Opti()  # Optimization problem
    W = opti.variable(3, N + 1)
    X = W[0:2, :]
    U = W[2, :]
    T = 1  # final time
    dt = T / N
    opti.minimize(0.5 * dt * cs.sum2(U**2))
    f = lambda x, u: cs.vertcat(x[1], u)  # dx/dt = f(x,u)
    F = cs.hcat([f(X[:, _], U[:, _]) for _ in range(N)])  # forward euler
    opti.subject_to(
        opti.bounded(
            0,
            cs.vec(
                cs.hcat(
                    [
                        X[:, 1 : N + 1] - X[:, 0:N] - dt * F,
                        X[:, 0],
                        X[:, -1] - cs.DM([1, 0]),
                    ]
                )
            ),
            0,
        )
    )
    opti.set_initial(opti.x, cs.DM.ones(opti.nx, 1))
    if solve:
        opti.solver("ipopt", {"ipopt.print_level": 3})
        sol = opti.solve()
        print(f"opti construction+ IPOPT solve time [s]: {pf() - t1}")
        ini = get_initial(opti=opti)
        ret = (opti, ini, sol)
    else:
        print(f"opti construction time [s]: {pf() - t1}")
        ini = get_initial(opti=opti)
        ret = (opti, ini, None)

    return ret


@register
def racecar_ocp(n=200, solve=False) -> tuple:
    """
    racecar problem from casadi repo.

    direct transcription -> NLP

    Optimal objective : 1.9 (approx)

    Returns
    -------
    tuple
        (opti,initial guess)
    """
    t1 = pf()
    N = n
    opti = cs.Opti()
    X = opti.variable(2, N + 1)
    pos = X[0, :]
    speed = X[1, :]
    U = opti.variable(1, N)
    T = opti.variable()
    opti.minimize(T)
    xs = cs.SX.sym("xs", 2, 1)
    us = cs.SX.sym("us", 1, 1)
    ts = cs.SX.sym("us", 1, 1)
    f = cs.Function("f", [xs, us], [cs.vertcat(xs[1], us - xs[1])])
    dt = T / N

    k1 = f(xs, us)
    k2 = f(xs + ts / 2 * k1, us)
    k3 = f(xs + ts / 2 * k2, us)
    k4 = f(xs + ts * k3, us)
    x_next = xs + ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next = cs.Function("x_next", [xs, us, ts], [x_next]).map(N, "serial")

    opti.subject_to(X[:, 1:] == x_next(X[:, 0:-1], U, dt))

    limit = cs.Function("limit", [us], [1 - 0.5 * cs.sin(2 * cs.pi * us)]).map(
        N + 1, "serial"
    )
    opti.subject_to(speed <= limit(pos))
    opti.subject_to(opti.bounded(0, U, 1))
    opti.subject_to(pos[0] == 0)
    opti.subject_to(speed[0] == 0)
    opti.subject_to(pos[-1] == 1)
    opti.subject_to(T >= 0)
    opti.set_initial(opti.x, 1 * cs.DM.ones(opti.nx, 1))
    if solve:
        opti.solver("ipopt", {"ipopt.print_level": 3})
        sol = opti.solve()
        print(f"opti construction+ IPOPT solve time [s]: {pf() - t1}")
        ini = get_initial(opti=opti)
        ret = (opti, ini, sol)
    else:
        print(f"opti construction time [s]: {pf() - t1}")
        ini = get_initial(opti=opti)
        ret = (opti, ini, None)

    return ret


@register
def hs015_nlp(solve=False) -> tuple:
    # optimal value -306.5
    t1 = pf()
    opti = cs.Opti()
    x = opti.variable()
    y = opti.variable()
    objective = 100 * (y - x**2) ** 2 + (1 - x) ** 2
    opti.minimize(objective)
    opti.subject_to(x * y >= 1)
    opti.subject_to(x + y**2 >= 0)
    opti.subject_to(x <= 0.5)
    opti.set_initial(x, -2)
    opti.set_initial(y, 1)
    if solve:
        opti.solver("ipopt", {"ipopt.print_level": 3})
        sol = opti.solve()
        print(f"opti construction+ IPOPT solve time [s]: {pf() - t1}")
        ini = get_initial(opti=opti)
        ret = (opti, ini, sol)
    else:
        print(f"opti construction time [s]: {pf() - t1}")
        ini = get_initial(opti=opti)
        ret = (opti, ini, None)

    return ret
