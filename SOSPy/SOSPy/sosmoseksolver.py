from mosek.fusion import *
import mosek
import numpy as np
import sys
import time
from scipy.sparse import csc_matrix, csr_matrix

def sosmoseksolver(At:csr_matrix, b:csc_matrix, c:csr_matrix, K:dict, verbose:int=1) -> tuple[list, list, dict]:
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

    # verbose = 1 default is 1, print out the information
    At_temp = At.toarray()
    c_temp = c.toarray()
    b_temp = b.toarray()

    with Model() as model:
        VAR_DIC = {}
        A_DIC = {}
        C_DIC = {}
        
        # Create Semidefinite variables
        for i in range(nsdpvar):   # Loop through Semidefinite Variable
            VAR_DIC[i] = model.variable(Domain.inPSDCone(K['s'][i]))
            A_DIC[i] = At_temp[Kindex[i]:Kindex[i+1]]
            C_DIC[i] = c_temp[Kindex[i]:Kindex[i+1]]
            
        # Create scalar variables, add to the end of VAR_DIC
        if nvars > 0:
            nidx = len(VAR_DIC)
            VAR_DIC[nidx] = model.variable(nvars)
            A_DIC[nidx] = At_temp[:nvars]
            C_DIC[nidx] = c_temp[:nvars]
            
        
        # Objective function and constraints
        if len(VAR_DIC)>1:   # More than 1 item in objective/constraints function
            #OBJ_SUM = Expr.add(Expr.dot(C_DIC[0],VAR_DIC[0]),Expr.dot(C_DIC[1],VAR_DIC[1]))
            OBJ_SUM = Expr.add(Expr.mul(C_DIC[0].T,VAR_DIC[0].reshape(RSPind[0],1)),Expr.dot(C_DIC[1].T,VAR_DIC[1].reshape(RSPind[1],1)))
            CON_SUM = Expr.add(Expr.mul(A_DIC[0].T,VAR_DIC[0].reshape(RSPind[0],1)),Expr.mul(A_DIC[1].T,VAR_DIC[1].reshape(RSPind[1],1)))
            for i in range(2,len(VAR_DIC)):
                #OBJ_SUM = Expr.add(OBJ_SUM, Expr.dot(C_DIC[i],VAR_DIC[i]))
                OBJ_SUM = Expr.add(OBJ_SUM, Expr.mul(C_DIC[i].T,VAR_DIC[i].reshape(RSPind[i],1)))
                CON_SUM = Expr.add(CON_SUM, Expr.mul(A_DIC[i].T,VAR_DIC[i].reshape(RSPind[i],1)))
        else:
            #OBJ_SUM = Expr.dot(C_DIC[0], VAR_DIC[0])
            OBJ_SUM = Expr.mul(C_DIC[0].T, VAR_DIC[0].reshape(RSPind[0],1))
            CON_SUM = Expr.mul(A_DIC[0].T, VAR_DIC[0].reshape(RSPind[0],1))
            
        # Objective function
        obj = model.objective(ObjectiveSense.Minimize, OBJ_SUM)
        
        # Constraints
        constr = model.constraint(Expr.sub(CON_SUM, b_temp), Domain.equalsTo(0.0))
        
        if verbose == 1:
            model.setLogHandler(sys.stdout)
        # Solve the model
        
        model.solve()
        end_time = time.time()

        # Get status information about the solution
        feasbl = model.getProblemStatus()
        if feasbl != ProblemStatus.PrimalAndDualFeasible:
            sol = []
            y = np.array([])
        else:
            sol = []
            y = constr.dual()
            if nvars > 0:
                sol.append(VAR_DIC[nidx].level())
                nidx = 1
            else:
                nidx = 0
            for i in range(len(VAR_DIC)-nidx):
                sol.append(VAR_DIC[i].level())

        task = model.getTask()
        info = {}
        info['cpusec'] = round(end_time-start_time,5)
        info['iter'] = task.getintinf(mosek.iinfitem.intpnt_iter)
        info['feasratio'] = round(task.getdouinf(mosek.dinfitem.intpnt_opt_status),5)
        if feasbl == ProblemStatus.PrimalInfeasible:
            info['pinf'] = 1
        else:
            info['pinf'] = 0
        if feasbl == ProblemStatus.DualInfeasible:
            info['dinf'] = 1
        else:
            info['dinf'] = 0
        #info['numerr'] = 0

        # convert sol and y to list
        x = []
        for item in sol:
            x.extend(item.flatten().tolist())
        y = y.flatten().tolist()

    return x, y, info