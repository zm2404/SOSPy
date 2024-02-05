import numpy as np
import scs
from scipy.sparse import csc_matrix, csr_matrix
from scipy.linalg import block_diag
import time

from numpy.linalg import matrix_rank
from .removeredundantrow import remove_redundant_row


# The vec function as documented in api/cones
def vec(S:np.ndarray) -> np.ndarray:
    # The input S is 2-dimensional, the output is 1-dimensional
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]

# The mat function as documented in api/cones
def mat(s:np.ndarray, n:int) -> np.ndarray:
    # The input s is 1-dimensional, the output is 2-dimensional
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S

def restore_symmetric_matrix(upper_triangle_values:list, n:int) -> list:    
    matrix = [0]*(n**2)
    
    idx = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i*n+j] = upper_triangle_values[idx]
            matrix[j*n+i] = upper_triangle_values[idx] 
            idx += 1
    
    return matrix

def make_symmetric(vector:np.ndarray) -> np.ndarray:
    '''
    Since scs consider A as a symmetric matrix, and the function vec() consider A as symmetric by default, 
    we need to make A symmetric.
    '''
    # check if the vector length is a perfect square
    size = int(np.sqrt(len(vector)))
    if size * size != len(vector):
        raise ValueError("Vector length must be a perfect square")

    matrix = np.reshape(vector, (size, size))
    
    # make the matrix symmetric
    symmetric_matrix = (matrix + matrix.T) / 2
    
    return symmetric_matrix

def sosscssolver(At:csr_matrix, b:csc_matrix, c:csr_matrix, K:dict, options:dict={}, verbose:int=1) -> tuple[list, list, dict]:
    '''
    This function is to solve the optimization problem with SCS solver

    Output:
        x[list]: primal problem solution
        y[list]: dual problem solution
        info[dic]: information about the solution
    '''
    info = {}
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


    ############################### process b ##################################
    ncons = At.shape[1]  # number of constraints
    b_z = b.toarray().flatten()

    ############################# process A and c #############################
    A = []
    C = []
    A_psd = []
    totalpsd = 0
    LEN_VAR = [nvars]
    #### sdp variable
    for i in range(nsdpvar):
        C.append(vec(c[Kindex[i]:Kindex[i+1]].reshape(K['s'][i],K['s'][i]).toarray()))
        npsdvars = int(K['s'][i]*(K['s'][i]+1)/2)
        totalpsd = totalpsd + npsdvars
        LEN_VAR.append(totalpsd+nvars)
        A_psd.append(-1*np.eye(npsdvars))  # for psd cones
        A_temp = []
        for j in range(ncons):
            A_temp.append(vec(make_symmetric(At[Kindex[i]:Kindex[i+1],j].toarray().flatten())))
        A.append(np.vstack(A_temp))
        
    A_z = np.hstack(A)  # for zero cones
    c_scs = np.hstack(C)
    A_p = np.hstack((np.zeros((totalpsd,nvars)),block_diag(*A_psd)))    # for psd cones
    b_p = np.zeros(totalpsd)    # for psd cones

    #### scalar variable
    if nvars>0:
        c_scs = np.hstack((c[:nvars].toarray().flatten(), c_scs))
        A_z = np.hstack((At[:nvars,:].toarray().T, A_z))    # for zero cones
        

    ############################ Process A and b ###############################
    # Remove redundant rows
    if matrix_rank(A_z.T) < min(A_z.shape):
        old_Az = A_z
        A_z, b_z, rows_to_keep = remove_redundant_row(A_z, b.toarray())
        if A_z is None:
            info['cpusec'] = 0
            info['iter'] = 0
            info['status'] = 'infeasible'
            info['pinf'] = 1
            info['dinf'] = 1
            return [], [], info
        b_z = b_z.flatten()
        rows_to_keep += list(range(old_Az.shape[0],old_Az.shape[0]+A_p.shape[0]))
        info['rows_to_keep'] = rows_to_keep


    ############################### Collect A,b,c,P ###############################
    A_scs = csc_matrix(np.vstack((A_z,A_p)))    # for zero and psd cones
    P_scs = csc_matrix(np.zeros((A_scs.shape[1],A_scs.shape[1])))
    b_scs = np.hstack((b_z,b_p))    # for zero and psd cones

    ################################ collect dim ##################################
    if nsdpvar > 1:
        s_dim = K['s']
    else:
        s_dim = K['s'][0]

    ################################# options ####################################
    if 'max_iters' in options:
        max_iters = options['max_iters']
    else:
        max_iters = 100000

    if 'normalize' in options:
        normalize = options['normalize']
    else:
        normalize = True

    if 'scale' in options:
        scale = options['scale']
    else:
        scale = 0.1

    if 'adaptive_scale' in options:
        adaptive_scale = options['adaptive_scale']
    else:
        adaptive_scale = True
    
    if 'rho_x' in options:
        rho_x = options['rho_x']
    else:
        rho_x = 1.0e-6
    
    if 'eps_abs' in options:
        eps_abs = options['eps_abs']
    else:
        eps_abs = 1.0e-04

    if 'eps_rel' in options:
        eps_rel = options['eps_rel']
    else:
        eps_rel = 1.0e-04

    if 'eps_infeas' in options:
        eps_infeas = options['eps_infeas']
    else:
        eps_infeas = 1.0e-7

    if 'alpha' in options:
        alpha = options['alpha']
    else:
        alpha = 1.5

    if 'time_limit_secs' in options:
        time_limit_secs = options['time_limit_secs']
    else:
        time_limit_secs = 0

    if verbose == 1:
        verbose = True
    else:
        verbose = False

    if 'acceleration_lookback' in options:
        acceleration_lookback = options['acceleration_lookback']
    else:
        acceleration_lookback = 10

    if 'acceleration_interval' in options:
        acceleration_interval = options['acceleration_interval']
    else:
        acceleration_interval = 10    
    
    ################################# solve #######################################
    z_dim = b_z.shape[0]
    data = dict(P=P_scs, A=A_scs, b=b_scs, c=c_scs)
    cone = dict(z=z_dim,s=s_dim)
    # Initialize solver
    solver = scs.SCS(data,cone,eps_abs=eps_abs,eps_rel=eps_rel,max_iters=max_iters,normalize=normalize,scale=scale,adaptive_scale=adaptive_scale,rho_x=rho_x,eps_infeas=eps_infeas,alpha=alpha,time_limit_secs=time_limit_secs,verbose=verbose,acceleration_lookback=acceleration_lookback,acceleration_interval=acceleration_interval)
    #solver = scs.SCS(data,cone,eps_abs=eps_abs,eps_rel=eps_rel,max_iters=max_iters,normalize=normalize,scale=scale,adaptive_scale=adaptive_scale,rho_x=rho_x,eps_infeas=eps_infeas,alpha=alpha,time_limit_secs=time_limit_secs,verbose=verbose,acceleration_lookback=acceleration_lookback,acceleration_interval=acceleration_interval)
    #solver = scs.SCS(data,cone,eps_abs=1e-9,eps_rel=1e-9)
    # Solve
    start_time = time.time()
    sol = solver.solve()
    end_time = time.time()


    ################################# post process ################################
    sol_conv = []   # solution converted to standard form
    sol_conv.extend([1]*nvars) # number of scalar variables
    for i in range(nsdpvar):
        K_size = np.ones((K['s'][i],1))
        sol_conv.extend(vec(K_size @ K_size.T).tolist())

    sol_inv = np.linalg.inv(np.diag(sol_conv))    
    #sol_inv = np.diag(sol_conv)
    
    x_temp = [item for item in sol_inv @ sol['x']]
    x = []
    x.extend(x_temp[:nvars])
    for i in range(nsdpvar):
        x.extend(restore_symmetric_matrix(x_temp[LEN_VAR[i]:LEN_VAR[i+1]],K['s'][i]))

    #y = [item for item in sol['y'][:ncons]]
    y = [item for item in sol['y']]

    info = {}
    info['cpusec'] = round(end_time-start_time,5)
    info['iter'] = sol['info']['iter']
    info['status'] = sol['info']['status']
    info['pinf'] = round(sol['info']['res_pri'],5)
    info['dinf'] = round(sol['info']['res_dual'],5)
    info['A_scs'] = A_scs
    info['LEN_VAR'] = LEN_VAR
 
    return x, y, info