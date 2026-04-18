import numpy as np
import casadi as cs
import unopy
from typing import Any
import subprocess
#get unicode working in cmd
subprocess.run('chcp 65001',shell=True)

def opti2unomodel(opti:cs.Opti,x0:list)->Any:
    """
    Solves a CasADi Opti problem using the Uno solver Python interface.
    
    all constraints assumed to be nonlinear (no simple bounds in opti)
    bound vectors are constant

    min. f
    s.t  lbg<=g<=ubg
    """
    # extract decision vector objective constraint metadata
    x = opti.x
    f = opti.f
    g = opti.g
    nx = x.shape[0]
    ng = opti.ng

    #constants
    lbg = cs.evalf(opti.lbg).full().flatten().tolist()
    ubg = cs.evalf(opti.ubg).full().flatten().tolist()

    # casadi functions for numerical evaluation 
    lam_f=cs.MX.sym('lam_f')
    lag=opti.f*lam_f
    if opti.ng:
      lag=lag-opti.lam_g.T@opti.g
    # uno needs lower triangular part of hessian
    H, _ = cs.hessian(lag, opti.x)
    H_lower = cs.tril(H)
    # extract nonzero entries using indices
    H_sp = H_lower.sparsity()
    hessian_row_indices, hessian_column_indices = H_sp.get_triplet()
    num_hessian_nonzeros = H_sp.nnz()
    H_nz = H_lower[H_sp]
    H_func = cs.Function('H', [x, lam_f, opti.lam_g], [H_nz])
    # based on example for unopy types of arguments for these funcs. need clarity  
    def lagrangian_hessian(x_arr, obj_multiplier, mult_arr, hess_vals):
        if opti.ng==0:
            mult_arr = cs.DM([])
        h_val = np.array(H_func(x_arr, obj_multiplier, mult_arr).nonzeros()).flatten()
        hess_vals[:] = h_val
    # hessian vector product for Kyrlov methods (faster)
    grad_L = cs.gradient(lag, opti.x)
    v_sym = cs.MX.sym('v', opti.nx)
    h_v_prod = cs.jtimes(grad_L, opti.x, v_sym)
    hv_func = cs.Function('hv', [x, lam_f, opti.lam_g, v_sym], [h_v_prod])
    def lagrangian_hessian_operator(x_arr, obj_multiplier, mult_arr, v_arr, hv_vals):
        res = hv_func(x_arr, obj_multiplier, mult_arr, v_arr).full().flatten()
        hv_vals[:] = res    

    # casadi funcs. to unopy funcs.
    obj_func = cs.Function('obj', [x], [f])
    grad_func = cs.Function('grad', [x], [cs.gradient(f, x)])
    con_func = cs.Function('con', [x], [g])
    jac_func = cs.Function('jac_f', [x], [cs.jacobian(g, x)])

    # jacobian needs only nonzero entries
    # sniff it out using sparsity class
    jac_sparsity = jac_func.sparsity_out(0)
    num_jac_nonzeros = jac_sparsity.nnz()
    jac_row_indices, jac_col_indices = jac_sparsity.get_triplet()

    # unopy compatible functions
    def get_objective(x_arr):
        return float(obj_func(x_arr).full()[0, 0]) #float or ndarray?

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
        unopy.PROBLEM_NONLINEAR,  #api to specify type of optimization?
        nx,
        [-float('inf')] * nx,
        [float('inf')] * nx,
        unopy.ZERO_BASED_INDEXING
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
            get_jacobian
        )

    model.set_lagrangian_hessian(
            num_hessian_nonzeros,
            unopy.LOWER_TRIANGLE,
            hessian_row_indices,
            hessian_column_indices,
            lagrangian_hessian
        )
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_NEGATIVE)
    #model.set_lagrangian_hessian_operator(lagrangian_hessian_operator)

    #guess for decision vector
    model.set_initial_primal_iterate(x0)
    return model

def kelly()->tuple:
    '''
    kelly block problem

    direct transcription -> QP

    Optimal objective : 6

    Returns
    -------
    tuple
        (opti,initial guess)
    '''
    N =1000 # number of control intervals
    opti = cs.Opti() # Optimization problem
    W=opti.variable(3,N+1)
    X=W[0:2,:]
    U=W[2,:]
    T = 1      # final time
    dt=T/N
    opti.minimize(0.5*dt*cs.sum2(U**2))
    f = lambda x,u: cs.vertcat(x[1],u) # dx/dt = f(x,u)
    F=cs.hcat([f(X[:,_],U[:,_]) for _ in range(N)]) # forward euler
    opti.subject_to(opti.bounded(0,cs.vec(cs.hcat([X[:,1:N+1]-X[:,0:N]-dt*F,X[:,0],X[:,-1]-cs.DM([1,0])])),0))
    opti.set_initial(opti.x,cs.DM.ones(opti.nx,1))
    #opti.solver("ipopt",{'ipopt.hessian_approximation':'limited-memory'}) # set numerical backend
    #sol = opti.solve()
    return opti, opti.debug.value(opti.x,opti.initial()).flatten().tolist()

def racecar()->tuple:
    '''
    racecar problem from casadi repo.

    direct transcription -> NLP

    Optimal objective : 1.9 (approx)

    Returns
    -------
    tuple
        (opti,initial guess)
    '''
    N = 500
    opti = cs.Opti() 
    X = opti.variable(2,N+1)
    pos   = X[0,:]
    speed = X[1,:]
    U = opti.variable(1,N)  
    T = opti.variable()      
    opti.minimize(T) 
    f = lambda x,u: cs.vertcat(x[1],u-x[1]) 
    dt = T/N 
    for k in range(N): 
       k1 = f(X[:,k],         U[:,k])
       k2 = f(X[:,k]+dt/2*k1, U[:,k])
       k3 = f(X[:,k]+dt/2*k2, U[:,k])
       k4 = f(X[:,k]+dt*k3,   U[:,k])
       x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
       opti.subject_to(X[:,k+1]==x_next) 

    limit = lambda pos: 1-cs.sin(2*cs.pi*pos)/2
    opti.subject_to(speed<=limit(pos))   
    opti.subject_to(opti.bounded(0,U,1)) 
    opti.subject_to(pos[0]==0)  
    opti.subject_to(speed[0]==0)  
    opti.subject_to(pos[-1]==1)  
    opti.subject_to(T>=0) 
    opti.set_initial(speed, 1)
    opti.set_initial(T, 1)
    #opti.solver("ipopt",{'ipopt.hessian_approximation':'limited-memory'}) # set numerical backend
    #sol = opti.solve()
    return opti, opti.debug.value(opti.x,opti.initial()).flatten().tolist()


if __name__ == '__main__':
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
