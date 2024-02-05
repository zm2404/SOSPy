import numpy as np
import math
from scipy.sparse import csr_matrix, lil_matrix, vstack, eye, kron, hstack, issparse, csc_matrix, coo_matrix
import importlib
#import cvxpy as cp

from .monomials import monomials
from .inconvhull import inconvhull
from .getconstraint import getconstraint
from .findcommonZ import findcommonZ
from .spantiblkdiag import spantiblkdiag
from .spblkdiag import spblkdiag
from .sparsemultipart import sparsemultipart
from .sosmoseksolver import sosmoseksolver
from .sosscssolver import sosscssolver, mat
from .soscvxoptsolver import soscvxoptsolver
from .sosprogram import sosprogram


def sparse_fliplr(A:csr_matrix) -> csr_matrix:
    A_coo = A.tocoo()
    flipped_col = A.shape[1] - A_coo.col - 1
    flipped_A = coo_matrix((A_coo.data, (A_coo.row, flipped_col)), shape=A_coo.shape)
    return flipped_A.tocsr()

def sparse_flipud(A:csr_matrix) -> csr_matrix:
    A_coo = A.tocoo()
    flipped_row = A.shape[0] - A_coo.row - 1
    flipped_A = coo_matrix((A_coo.data, (flipped_row, A_coo.col)), shape=A_coo.shape)
    return flipped_A.tocsr()

def addextrasosvar(sos:sosprogram, I:list[int]) -> tuple[sosprogram, int]:
    feasextrasos = 1
    for i in I:
        numstates = sos.expr['Z'][i].shape[1]
        maxdeg = int(np.max(np.sum(sos.expr['Z'][i], axis=1)))  # maximum total degree of the monomials
        mindeg = int(np.min(np.sum(sos.expr['Z'][i], axis=1)))  # minimum total degree of the monomials
        Z = monomials(numstates, range(math.floor(mindeg/2), math.ceil(maxdeg/2) + 1))

        # Discarding unnecessary monomials
        maxdegree = np.max(sos.expr['Z'][i], axis=0) / 2    # since sos.expr['Z'][i] is a sparse matrix, maxdegree is a 1D array
        mindegree = np.min(sos.expr['Z'][i], axis=0) / 2    # since sos.expr['Z'][i] is a sparse matrix, mindegree is a 1D array

        Zdummy1 = maxdegree.toarray() - Z.toarray()   # use toarray() to convert a sparse matrix to a dense matrix so that we can do element-wise subtraction
        Zdummy2 = Z.toarray() - mindegree.toarray()
        I, _ = np.where(np.concatenate((Zdummy1, Zdummy2), axis=1) < 0) # find the rows with negative entries
        IND =  np.setdiff1d(np.arange(Z.shape[0]), I)   # find the rows with non-negative entries
        if len(IND) == 0:
            print("\n Warning: Inequality constraint {} in your sos program structure corresponds to a polynomial that is not sum-of-squares.\n".format(i))
            feasextrasos = 0
            return
        else:
            Z = Z[IND, :]   # Only keep the rows with non-negative entries


        if sos.expr['type'][i] == 'sparse':
           Z2 = sos.expr['Z'][i]/2
           Z = inconvhull(Z,Z2)
           Z = csr_matrix(Z)
        
        if sos.expr['type'][i] == 'sparsemultipartite':
            Z2 = sos.expr['Z'][i]/2
            info2 = sos.expr['multipart'][i]
            sizeinfo2m = len(info2)
            vecindex = []
            for indm in range(sizeinfo2m):
                vecindex.append([])
                for indn in range(len(info2[indm])):
                    varcheckindex = [i for i, y in enumerate(sos.symvartable) if info2[indm][indn] == y] # Assume info2[indm][indn] is a symbol
                    if varcheckindex:
                        vecindex[indm].extend(varcheckindex)
                    else:
                        vecindex[indm].extend([len(info2[0]) + [i for i, y in enumerate(sos.varmat['symvartable']) if info2[indm][indn] == y][0]])
            
            Z = sparsemultipart(Z,Z2,vecindex)
            Z = csr_matrix(Z)
        # Z in the quadratic representation has now been updated.

        dimp = sos.expr['b'][i].shape[1]   # detecting whether Mineq is active

        # Adding slack variables
        sos.extravar['num'] += 1
        var = sos.extravar['num'] - 1
        sos.extravar['Z'][var] = csr_matrix(Z)
        [T,ZZ] = getconstraint(Z)
        sos.extravar['ZZ'][var] = csr_matrix(ZZ)
        sos.extravar['T'][var] = csr_matrix(T.T)

        sos.extravar['idx'][var+1] = sos.extravar['idx'][var] + (Z.shape[0] * dimp) ** 2
        for j in range(sos.expr['num']):
            sos.expr['At'][j] = vstack([sos.expr['At'][j], csr_matrix((sos.extravar['T'][var].shape[0]*dimp**2, sos.expr['At'][j].shape[1]))])


        if dimp == 1:
            pc = {}
            pc['Z'] = sos.extravar['ZZ'][var]
            pc['F'] = -eye(pc['Z'].shape[0])
            R1, R2, newZ = findcommonZ(sos.expr['Z'][i], pc['Z'])

            if sos.expr['At'][i].size == 0:
                sos.expr['At'][i] = csr_matrix((sos.expr['At'][i].shape[0], R1.shape[0]))
            
            sos.expr['At'][i] = (sos.expr['At'][i] @ R1).tolil()
            lidx = sos.extravar['idx'][var]
            uidx = sos.extravar['idx'][var+1]
            sos.expr['At'][i][lidx:uidx,:] = sos.expr['At'][i][lidx:uidx,:] - sos.extravar['T'][var] @ pc['F'] @ R2
            sos.expr['At'][i] = sos.expr['At'][i].tocsr()
            sos.expr['b'][i] = R1.T @ sos.expr['b'][i]
            sos.expr['Z'][i] = newZ

        else:
            ZZ = sparse_flipud(ZZ)
            T = sparse_flipud(T)
            Zcheck = sos.expr['Z'][i]
            R1, R2, Znew = findcommonZ(Zcheck, ZZ)

            R1 = sparse_fliplr(R1)
            R2 = sparse_fliplr(R2)
            Znew = sparse_flipud(Znew)

            R1sum = R1.sum(axis=0).tolist()[0]
            T = R2.T @ T

            ii = 0
            #sig_ZZ = ZZ.shape[0]
            sig_Z = Z.shape[0]
            sig_Znew = Znew.shape[0]

            Tf = lil_matrix((dimp**2 * sig_Znew, (dimp * sig_Z)**2))
            Sv = lil_matrix((sig_Znew * dimp**2, 1))
            for j in range(sig_Znew):
                Mt0 = lil_matrix((dimp, dimp * sig_Z**2))
                for k in range(sig_Z):
                    Mt0[:, (dimp*sig_Z)*k: (dimp*sig_Z)*(k+1)] = kron(eye(dimp), T[j, sig_Z*k: sig_Z*(k+1)])
                Tf[j*dimp**2:(j+1)*dimp**2, :] = kron(eye(dimp), Mt0)

                if R1sum[j] == 1:
                    b_temp = sos.expr['b'][i].tolil()
                    Sv[j*dimp**2:(j+1)*dimp**2] = np.reshape(b_temp[dimp*ii:dimp*(ii+1), :].T, (1, dimp**2)).T
                    if ii < Zcheck.shape[0]-1:
                        ii += 1
                else:
                    # sparse matrix slicing, need to be careful. probably need to convert to csc matrix
                    sos.expr['At'][i] = hstack([sos.expr['At'][i][:,:j*dimp**2], csr_matrix((sos.expr['At'][i].shape[0], dimp**2)), sos.expr['At'][i][:,j*dimp**2:]])

            lidx = sos.extravar['idx'][var]
            uidx = sos.extravar['idx'][var+1]
            sos.expr['At'][i] = sos.expr['At'][i].tolil()
            sos.expr['At'][i][lidx:uidx,:] = Tf.tocsr().transpose()
            sos.expr['At'][i] = sos.expr['At'][i].tocsr()
            sos.expr['b'][i] = Sv.tocsr()

    return sos, feasextrasos


def addextrasosvar2(sos:sosprogram, I:list[int]) -> sosprogram:
    numstates = sos.expr['Z'][0].shape[1]
    for i in I:
        # Create extra variables
        maxdeg = int(np.max(np.sum(sos.expr['Z'][i], axis=1)))
        mindeg = int(np.min(np.sum(sos.expr['Z'][i], axis=1)))
        Z = monomials(numstates, range(math.floor(mindeg/2), math.ceil(maxdeg/2) + 1))

        # Discarding unnecessary monomials
        maxdegree = sos.expr['Z'][i].max(axis=0) / 2
        mindegree = sos.expr['Z'][i].min(axis=0) / 2
        j = 0
        while j < Z.shape[0]:
            Zdummy1 = maxdegree.toarray() - Z[j].toarray()
            Zdummy2 = Z[j].toarray() - mindegree.toarray()
            idx, _ = np.where(np.concatenate((Zdummy1, Zdummy2), axis=1) < 0)
            if idx.size > 0:
                Z = np.vstack([Z[:j], Z[j+1:]])
            else:
                j += 1
            
        # Add the variables
        for k in range(2):
            sos.extravar['num'] += 1
            var = sos.extravar['num'] - 1
            sos.extravar['Z'][var] = csr_matrix(Z)
            T,ZZ = getconstraint(Z)
            sos.extravar['ZZ'][var] = ZZ    
            sos.extravar['T'][var] = T.T
            sos.extravar['idx'][var+1] = sos.extravar['idx'][var] + Z.shape[0]**2
            for j in range(sos.expr['num']):
                sos.expr['At'][j] = vstack([sos.expr['At'][j], csr_matrix((sos.extravar['T'][var].shape[0], sos.expr['At'][j].shape[1]))])

            # Modifying expression
            degoffset = hstack([k, csr_matrix((1,numstates-1))])
            pc = {}
            pc['Z'] = sos.extravar['ZZ'][var] + vstack([degoffset]*sos.extravar['ZZ'][var].shape[0])
            pc['F'] = -eye(pc['Z'].shape[0])
            R1, R2, newZ = findcommonZ(sos.expr['Z'][i], pc['Z'])

            if sos.expr['At'][i].size == 0:
                sos.expr['At'][i] = csr_matrix((sos.expr['At'][i].shape[0], R1.shape[0]))

            sos.expr['At'][i] = sos.expr['At'][i] @ R1
            lidx = sos.extravar['idx'][var]
            uidx = sos.extravar['idx'][var+1]
            sos.expr['At'][i] = sos.expr['At'][i].tolil()
            sos.expr['At'][i][lidx:uidx,:] = sos.expr['At'][i][lidx:uidx,:] - sos.extravar['T'][var] @ pc['F'] @ R2
            sos.expr['At'][i] = sos.expr['At'][i].tocsr()
            sos.expr['b'][i] = R1.T @ sos.expr['b'][i]
            sos.expr['Z'][i] = newZ

            Z = csr_matrix(Z[Z.toarray()<maxdegree.data[0]].reshape(-1,1))

    return sos


def processvars(sos:sosprogram, Atf:csr_matrix, bf:csc_matrix) -> tuple[csr_matrix, csc_matrix, dict, coo_matrix]:
    # Processing all variables
    # Decision variables
    K = {}
    K['s'] = []
    K['f'] = sos.var['idx'][0]
    RR = eye(K['f'])

    # Polynomial and SOS variables
    for i in range(sos.var['num']):
        if sos.var['type'][i] == 'poly':
            sizeX = int(sos.var['idx'][i+1] - sos.var['idx'][i])
            RR = spantiblkdiag(RR, eye(sizeX))
            K['f'] = sizeX + K['f']
        elif sos.var['type'][i] == 'sos':
            sizeX = int(math.sqrt(sos.var['idx'][i+1] - sos.var['idx'][i]))
            RR = spblkdiag(RR, eye(sizeX**2))
            K['s'].append(sizeX)

    for i in range(sos.extravar['num']):
        sizeX = int(math.sqrt(sos.extravar['idx'][i+1] - sos.extravar['idx'][i]))
        RR = spblkdiag(RR, eye(sizeX**2))
        K['s'].append(sizeX)

    At = RR.T @ Atf

    b = bf

    return At, b, K, RR



def sossolve(sos:sosprogram, options:dict=None, verbose:int=1) -> sosprogram:
    '''
    SOSSOLVE --- Solve a sum of squares program.

    SOSP = sossolve(SOSP)

    SOSP is the SOS program to be solved.

    Alternatively, SOSP = sossolve(SOSP,SOLVER_OPT) also defines the solver
    and/or the solver-specific options respectively by fields

    SOLVER_OPT.solver (name of the solver). This can be 'cvxopt,mosek,scs,copt,sdpa'.
    SOLVER_OPT.params (a structure containing solver-specific parameters)

    The default values for solvers is 'cvxopt'.
    '''
    # SOLVERS = ['MOSEK', 'CVXOPT', 'SCS', 'COPT', 'SDPA']
    # INSTALLED_SOLVERS = cp.installed_solvers()
    # ALL_SOLVER = [item for item in SOLVERS if item in INSTALLED_SOLVERS]

    own_mosek=0     # check if you have mosek license
    if importlib.util.find_spec('mosek') is not None:
        own_mosek=1

    # Default options
    if options is None:
        options = {}
        options['solver'] = 'cvxopt'
    else:
        if not 'solver' in options:
            options['solver'] = 'cvxopt'


    # Check if the SOS program is already solved
    if len(sos.solinfo['x']) != 0:
        raise ValueError('The SOS program is already solved.')
    
    # Adding slack variables to inequalities
    sos.extravar['idx'][0] = sos.var['idx'][sos.var['num']]
    # SOS variables
    feasextrasos = 1   # Will be set to 0 if constraints are immediately found infeasible
    I = [i for i, t in sos.expr['type'].items() if t in {'ineq','sparse','sparsemultipartite'}]
    if I:
        sos, feasextrasos = addextrasosvar(sos, I)
    # SOS variables type II (restrictedd on interval)
    I = [i for i, t in sos.expr['type'].items() if t == 'posint']
    if I:
        sos = addextrasosvar2(sos, I)

    # Processing all expressions
    Atf = hstack([sos.expr['At'][i] for i in range(sos.expr['num'])])
    bf = vstack([sos.expr['b'][i] for i in range(sos.expr['num'])])

    # Processing all variables
    At, b, K, RR = processvars(sos, Atf, bf)

    # Objective function
    c = lil_matrix((At.shape[0],1))

    indend = len(sos.var['idx'])
    indend_value = sos.var['idx'][indend-1]
    if issparse(sos.objective):
        objdim = sos.objective.shape[0]
    else:
        objdim = len(sos.objective)
    if objdim == 0:
        sos.objective = np.zeros(c[:indend_value].shape)
    
    c[:indend_value] = c[:indend_value] + sos.objective
    c = RR.T @ c

    feassosp = 1

    # if the addextrasosvar return infeasible solution.
    # If the problem is clearly infeasible, sedumi can return error
    # Return no solution if the problem is clearly infeasible from
    # addextrasosvar.
    if feassosp==0 or feasextrasos==0:
        print('Warning: Primal program infeasible, no solution produced.')
        info = {}
        info['iter'] = 0
        info['feasratio'] = -1
        info['pinf'] = 1
        info['dinf'] = 1
        info['cpusec'] = 0
        sos.solinfo['info'] = info
        sos.solinfo['solverOptions'] = options
        return sos
    
    
    ##################################### Solve the problem #####################################
    try:
        # SOLVER = None
        if options['solver'].lower() == 'mosek' and own_mosek == 1:
            x, y, info = sosmoseksolver(At, b, c, K, verbose)
        elif options['solver'].lower() == 'scs':
            x, y, info = sosscssolver(At, b, c, K, options, verbose)
        elif options['solver'].lower() == 'cvxopt':
            x, y, info = soscvxoptsolver(At, b, c, K, options, verbose)
        else:
            print('Warning: The solver you specified is not supported.')
    except:
        print('Warning: Cannot construct the problem.')
        info = {}
        info['iter'] = 0
        info['feasratio'] = -1
        info['status'] = 'infeasible'
        info['pinf'] = 1
        info['dinf'] = 1
        info['cpusec'] = 0
        sos.solinfo['info'] = info
        sos.solinfo['solverOptions'] = options
        return sos



    ################################### record the solution ###################################
    sos.solinfo['x'] = x
    sos.solinfo['y'] = y

    # 
    if options['solver'] == 'mosek' and own_mosek == 1:
        if not x and info['pinf']:
            print('Warning: Primal problem infeasible, no solution produced.')
            info['iter'] = 0
            info['feasratio'] = -1
            info['pinf'] = 1
            info['dinf'] = 1
            info['cpusec'] = 0
            sos.solinfo['info'] = info
            sos.solinfo['solverOptions'] = options
            return sos
    else:
        if info['status'] == 'infeasible':
            print('Warning: Problem infeasible, no solution produced.')
            info['iter'] = 0
            info['feasratio'] = -1
            info['pinf'] = 1
            info['dinf'] = 1
            info['cpusec'] = 0
            sos.solinfo['info'] = info
            sos.solinfo['solverOptions'] = options
            return sos
    

    print('\n','Residual norm', np.linalg.norm((At.T.dot(csr_matrix(x).T)-b).toarray().flatten()))

    sos.solinfo['RRx'] = (RR @ csc_matrix(x).T).toarray().flatten().tolist()
    if 'rows_to_keep' in info and options['solver'] == 'cvxopt':
        At_red = At[:, info['rows_to_keep']]
        sos.solinfo['RRy'] = (RR @ (c - At_red @ csc_matrix(y).T)).toarray().flatten().tolist()
        info.pop('rows_to_keep')
    elif 'A_scs' in info and options['solver'] == 'scs':
        A_y = info['A_scs'].T * csc_matrix(y).T  ## efficiency can be improved
        Ay_rec = []
        LEN_VAR = info['LEN_VAR']
        Ay_rec.append(A_y[:LEN_VAR[0]])
        for i in range(len(K['s'])):
            Ay_temp = mat(A_y[LEN_VAR[i]:LEN_VAR[i+1]].toarray().T, K['s'][i])
            Ay_rec.append(csc_matrix(Ay_temp.reshape(-1,1)))
        Ay = vstack(Ay_rec)
        sos.solinfo['RRy'] = (RR @ (c - Ay)).toarray().flatten().tolist()
        info.pop('A_scs')
        info.pop('LEN_VAR')
    else:
        sos.solinfo['RRy'] = (RR @ (c - At @ csc_matrix(y).T)).toarray().flatten().tolist()
    sos.solinfo['info'] = info
    sos.solinfo['solverOptions'] = options

    # print info
    for key, value in info.items():
        print(f'{key}: {value}')

    # Constructing the (primal and dual) solution vectors and matrices
    # If you want to have them, comment/delete the return command above.
    # In the future version, these primal and dual solutions will be computed only
    # when they are needed. We don't want to store redundant info.
    var = {}
    primal = {}
    dual = {}
    for i in range(sos.var['num']):
        if sos.var['type'][i] == 'poly':
            primal[i] = sos.solinfo['RRx'][sos.var['idx'][i]:sos.var['idx'][i+1]]
            dual[i] = sos.solinfo['RRy'][sos.var['idx'][i]:sos.var['idx'][i+1]]
        elif sos.var['type'][i] == 'sos':
            primaltemp = sos.solinfo['RRx'][sos.var['idx'][i]:sos.var['idx'][i+1]]
            dualtemp = sos.solinfo['RRy'][sos.var['idx'][i]:sos.var['idx'][i+1]]
            primal[i] = np.array(primaltemp).reshape(int(math.sqrt(len(primaltemp))),int(math.sqrt(len(primaltemp)))).T
            dual[i] = np.array(dualtemp).reshape(int(math.sqrt(len(dualtemp))),int(math.sqrt(len(dualtemp)))).T
    var['primal'] = primal
    var['dual'] = dual
    sos.solinfo['var'] = var
    
    extravar = {}
    primal = {}
    dual = {}
    for i in range(sos.extravar['num']):
        primaltemp = sos.solinfo['RRx'][sos.extravar['idx'][i]:sos.extravar['idx'][i+1]]
        dualtemp = sos.solinfo['RRy'][sos.extravar['idx'][i]:sos.extravar['idx'][i+1]]
        primal[i] = np.array(primaltemp).reshape(int(math.sqrt(len(primaltemp))),int(math.sqrt(len(primaltemp)))).T
        dual[i] = np.array(dualtemp).reshape(int(math.sqrt(len(dualtemp))),int(math.sqrt(len(dualtemp)))).T
    extravar['primal'] = primal
    extravar['dual'] = dual
    sos.solinfo['extravar'] = extravar

    decvar = {}
    decvar['primal'] = sos.solinfo['RRx'][:sos.var['idx'][0]]
    decvar['dual'] = sos.solinfo['RRy'][:sos.var['idx'][0]]
    sos.solinfo['decvar'] = decvar

    return sos
