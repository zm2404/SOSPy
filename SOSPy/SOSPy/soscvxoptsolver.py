from cvxopt import matrix, spmatrix, solvers
import numpy as np
import math
from scipy.sparse import tril, triu, csr_matrix, csc_matrix
import time

from numpy.linalg import matrix_rank
from .removeredundantrow import remove_redundant_row

def generate_Gs(indn:int,K:list,len_bef:int,total_len:int):
    '''
    indn is the index of PSD variable
    n = K['s'][indn] is the size of this PSD variable
    len_bef is the total length of variables before this PSD variable
    total_len is the total length of this Gs, added with other PSD variables
    '''
    n = K['s'][indn]  # the size
    val = []
    indi = []
    indj = []
    row_ind = 0
    for i in range(n):
        for j in range(i, n):
            if i != j:
                val.extend([-1.0]*2)
                indi.extend([row_ind,row_ind])
                indj.extend([i*n+j,j*n+i])
            else:
                val.extend([-1.0])
                indi.extend([row_ind])
                indj.extend([i*n+j])
            row_ind += 1
        
    for i in range(len(indi)):
        indi[i] = int(indi[i]+len_bef)
        
    Gs = spmatrix(val,indi,indj,(total_len,int(n**2))).T
    
    return Gs


def simplify_A(A:csc_matrix) -> list[list]:
    '''
    The row vectors in A are the flatten symmtric matrix.
    This function simplifies A, for example: 
    when n =3
        |a1  a2  a3|          |a1  2a2 2a3|
    A = |a2  a4  a5|, new_A = |2a2 a4  2a5|,
        |a3  a5  a6|          |2a3 2a5 a6 |
    
    and return the upper triangle of new_A matrix.
    
    return list [[a1,2a2,2a3,a4,2a5,a6],...]
    '''
    
    n = int(math.sqrt(A.shape[1]))
    new_A = []
    for i in range(A.shape[0]):
        A_tril = tril(A[i].reshape(n,n),-1)
        A_simp = A[i].reshape(n,n) + A_tril.transpose()
        new_A.append(triu(A_simp).toarray()[np.triu_indices(n)].tolist())

    return new_A


def restore_symmetric_matrix(upper_triangle_values:list,n:int) -> list:
    matrix = [0]*(n**2)
    
    idx = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i*n+j] = upper_triangle_values[idx]
            matrix[j*n+i] = upper_triangle_values[idx] 
            idx += 1
    
    return matrix


def soscvxoptsolver(At:csr_matrix, b:csc_matrix, c:csr_matrix, K:dict, options:dict={}, verbose:int=1) -> tuple[list, list, dict]:
    '''
    This function is to solve the optimization problem with CVXOPT solver

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

    nsdpvar = len(K['s'])   # Number of Semidefinite Program Variables
    nvars = Kindex[0]       # Number of scalar variable

    A_DIC = {}
    C_DIC = {}
    len_bef = []

    ###################### Construct A and C ######################
    # Create Scalar variables
    dic_len = 0
    if nvars > 0:
        A_DIC[0] = At[:nvars].toarray().T.tolist()
        C_DIC[0] = c[:nvars].toarray().T.tolist()
        dic_len += 1
        len_bef.append(len(A_DIC[0][0]))
    else:
        len_bef.append(0)

    # Create Semidefinite variables
    for i in range(nsdpvar):
        A_DIC[dic_len+i] = simplify_A(At[Kindex[i]:Kindex[i+1]].T)
        C_DIC[dic_len+i] = simplify_A(c[Kindex[i]:Kindex[i+1]].T)
        len_bef.append(len(A_DIC[dic_len+i][0]))

    # Construct A and C
    A_len = len(A_DIC)
    A = []
    for i in range(At.shape[1]):
        A_sum = []
        for j in range(A_len):
            A_sum += A_DIC[j][i]
        A.append(A_sum)
    C_sum = []
    for j in range(A_len):
        C_sum += C_DIC[j][0]

    # Remove redundant rows
    if matrix_rank(np.array(A).T) < min(np.array(A).shape):
        A, b, rows_to_keep = remove_redundant_row(np.array(A), b.toarray())
        if A is None:
            info['cpusec'] = 0
            info['iter'] = 0
            info['status'] = 'infeasible'
            info['pinf'] = 1
            info['dinf'] = 1
            return [], [], info
        A = matrix(A)
        b = matrix(b)
        info['rows_to_keep'] = rows_to_keep
    else:
        A = matrix(A).T
        b = matrix(b.toarray().tolist()).T
        
    c = matrix(C_sum)


    ##################### Construct Gs and hs #####################
    LEN_BEF = np.cumsum(len_bef)
    Gs = []
    hs = []
    for i in range(nsdpvar):
        Gs += [generate_Gs(i,K,LEN_BEF[i],LEN_BEF[-1])]
        hs += [matrix([0.0]*Ks2[i],(K['s'][i],K['s'][i]))]

    ####################### Call CVXOPT solver #####################
    if verbose == 0:
        solvers.options['show_progress'] = False
    else:
        solvers.options['show_progress'] = True
    if 'maxiters' in options:
        solvers.options['maxiters'] = options['maxiters']
    if 'abstol'  in options:
        solvers.options['abstol'] = options['abstol']
    if 'reltol'  in options:
        solvers.options['reltol'] = options['reltol']
    if 'feastol' in options:
        solvers.options['feastol'] = options['feastol']
    if 'refinement' in options:
        solvers.options['refinement'] = options['refinement']


    start_time = time.time()
    sol = solvers.sdp(c=c,Gs=Gs,hs=hs,A=A,b=b)
    end_time = time.time()


    ####################### Process the result ######################
    x = []
    temp_x = [item for item in sol['x']]
    x.extend(temp_x[:nvars])    # Scalar variables
    for i in range(nsdpvar):    # Semidefinite variables
        x.extend(restore_symmetric_matrix(temp_x[LEN_BEF[i]:LEN_BEF[i+1]],K['s'][i]))
    
    y = [item for item in sol['y']]

    info['cpusec'] = round(end_time-start_time,5)
    info['iter'] = sol['iterations']
    info['status'] = sol['status']
    info['pinf'] = round(sol['primal infeasibility'],5)
    info['dinf'] = round(sol['dual infeasibility'],5)

    return x, y, info


