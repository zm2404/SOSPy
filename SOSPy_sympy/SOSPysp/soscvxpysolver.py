import cvxpy as cp
import numpy as np
import time


def soscvxpysolver(At,b,c,K,verbose=1,SOLVER='CVXOPT'):
    '''
    This function is to solve the optimization problem with MOSEK solver

    Output:
        x[list]: primal problem solution
        y[list]: dual problem solution
        info[dic]: information about the solution
    '''
    start_time = time.time()
    if 'q' in K:
        raise Exception('K.q and K.r are supposed to be empty')

    if 'f' not in K:
        K['f'] = []
    if 'l' not in K:
        K['l'] = []

    nf = int(np.sum(K['f']))
    nl = int(np.sum(K['l']))

    Ks2 = [int(item**2) for item in K['s']]
    Kindex = nf + nl + np.cumsum([0]+Ks2)
    RSPind = Ks2 + [nf]

    nsdpvar = len(K['s'])   # Number of Semidefinite Program Variables
    nvars = Kindex[0]       # Number of scalar variable

    A_DIC = []
    C_DIC = []
    VAR_DIC = []
    constraints = []

    ######################### Create Semidefinite variables #########################
    for i in range(nsdpvar):
        VAR = cp.Variable((K['s'][i],K['s'][i]), PSD=True)    # Create a PSD variable
        #constraints += [VAR >> 0]
        VAR_DIC.append(VAR)    # Add to the dictionary
        A = []
        A_temp = At[Kindex[i]:Kindex[i+1]].T
        for j in range(A_temp.shape[0]):
            A.append(cp.trace(A_temp[j].toarray().reshape(K['s'][i],K['s'][i]) @ VAR))  # Trace of matrix multiplication
        A_DIC.append(A)
        C_temp = c[Kindex[i]:Kindex[i+1]]
        C_DIC.append(cp.trace(C_temp.toarray().reshape(K['s'][i],K['s'][i]) @ VAR))  # Trace of matrix multiplication
        
    ################### Create scalar variables, add to the end of VAR_DIC ###################
    if nvars > 0:
        VAR = cp.Variable(nvars)    # Create scalar variables
        VAR_DIC.append(VAR)
        A = []
        A_temp = At[:nvars].T
        for i in range(A_temp.shape[0]):
            A.append(A_temp[i].toarray() @ VAR)
        A_DIC.append(A)
        C_temp = c[:nvars]
        C_DIC.append(C_temp.toarray().T @ VAR)
        
    ########################### Contruct objective function ############################
    OBJ_SUM = sum(C_DIC)

    ############################# Construct constriants ################################
    B = b.toarray().flatten().tolist()
    for i in range(b.shape[0]):
        CON = 0
        for j in range(len(A_DIC)):
            CON += A_DIC[j][i]
        constraints += [CON == B[i]]
        
    ########################### Solve the problem ######################################
    prob = cp.Problem(cp.Minimize(OBJ_SUM),constraints)
    if SOLVER == 'CVXOPT':
        prob.solve(solver = cp.CVXOPT, verbose=verbose)
    elif SOLVER == 'MOSEK':
        prob.solve(solver = cp.MOSEK, verbose=verbose)
    elif SOLVER == 'SCS':
        prob.solve(solver = cp.SCS, verbose=verbose)
    elif SOLVER == 'COPT':
        prob.solve(solver = cp.COPT, verbose=verbose)
    elif SOLVER == 'SDPA':
        prob.solve(solver = cp.SDPA, verbose=verbose)
    
    end_time = time.time()

    ################################# Process solution #################################
    # primal solution
    x = []
    if nvars > 0:
        x.extend(VAR_DIC[-1].value.tolist())
    for i in range(nsdpvar):
        x.extend(VAR_DIC[i].value.flatten().tolist())

    # dual solution
    y = []
    for i in range(len(B)):
        dual_value = constraints[i].dual_value
        if isinstance(dual_value, (float,int,np.float64)):
            y.append(dual_value)
        else:
            y.extend(constraints[i].dual_value)

    # calculate feasratio
    violations = 0
    for constraint in constraints:
        residual = constraint.residual
        violations += np.sum(residual[residual > 1e-5])

    feasratio = 1 - violations / len(constraints)

    # info
    info = {}
    info['cpusec'] = round(end_time-start_time,5)
    info['iter'] = prob.solver_stats.num_iters
    info['feasratio'] = feasratio
    info['status'] = prob.status

    return x,y,info



    
