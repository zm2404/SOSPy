from .getdegrees import getdegrees
from scipy.sparse import csr_matrix, eye, hstack, vstack, find
from .getconstraint import getconstraint
from sympy import symbols, Matrix, prod
import numpy as np

# This function is used when Type == 'sos'
def myAdecvarprod(A,decvar):
    rows, cols, _ = find(A)

    sympoly = Matrix.zeros(A.shape[1], 1)

    for col in range(A.shape[1]):
        idx = [row for row in rows[cols == col]]
        
        if idx:
            sympoly[col, 0] = sum(decvar[i] for i in idx)

    return sympoly


def sosvar(sos,Type,Zsym):
    '''
    SOSVAR --- Declare a new SOSP variable in an SOS program 

    sos, V = sosvar(SOSP,TYPE,Z)

    SOSP is the sum of squares program.
    VAR is the new SOSP variable. This variable is described by TYPE and a 
    vector of monomial Z in the following manner:
    TYPE = 'sos'  : the new SOSP variable is a sum of squares        --- Z' * Q * Z
    TYPE = 'poly' : the new SOSP variable is an ordinary polynomial  --- F * Z
    '''

    if Type != 'sos' and Type != 'poly':
        raise Exception('Unknown type.')
    
    Z = getdegrees(Zsym, sos.symvartable)   # Z is a expononet matrix of monomials

    # Add new variable
    sos.var['num'] += 1
    var = sos.var['num']-1
    sos.var['type'][var] = Type
    sos.var['Z'][var] = csr_matrix(Z)

    if Type == 'sos':
        T, ZZ = getconstraint(Z)
        sos.var['ZZ'][var] = ZZ
        sos.var['T'][var] = T.T
        sos.var['idx'][var+1] = sos.var['idx'][var] + (Z.shape[0])**2

    elif Type == 'poly':
        sos.var['ZZ'][var] = csr_matrix(Z)
        sos.var['T'][var] = eye(Z.shape[0])
        sos.var['idx'][var+1] = sos.var['idx'][var] + Z.shape[0]

    # Modify existing equations
    for i in range(sos.expr['num']):
        sos.expr['At'][i] = vstack([sos.expr['At'][i], csr_matrix((sos.var['T'][var].shape[0], sos.expr['At'][i].shape[1]))])\
    
    # Modify existing objective
    if isinstance(sos.objective, list):
        sos.objective = csr_matrix((sos.var['idx'][var+1]-sos.var['idx'][var], 1))
    else:
        sos.objective = vstack([sos.objective, csr_matrix((sos.var['idx'][var+1]-sos.var['idx'][var], 1))])

    # Modify decision variable table
    varidx = sos.var['idx'][var+1]-sos.var['idx'][0]
    decvartabletempxxx = symbols(f'coeff_:{varidx}')    # coeff_0, coeff_1, ...
    decvartabletempxxx = decvartabletempxxx[sos.var['idx'][var]-sos.var['idx'][0]:] # only extract necessary coefficients

    for i in decvartabletempxxx:
        sos.symdecvartable.append(i)    # sympy symbols list

    sos.decvartable = str(sos.symdecvartable)   # sympy symbols list to string

    if Type == 'sos':
        if sos.var['ZZ'][var].shape[0] <= 1:
            vartable = sos.symvartable  # sympy symbols list
            temp_ZZ = sos.var['ZZ'][var].toarray()
            ZZSym = []
            for row in temp_ZZ:
                temp_res = [var**deg for var, deg in zip(vartable, row)]
                ZZSym.append(prod(temp_res))
            coeff = myAdecvarprod(sos.var['T'][var], sos.symdecvartable[sos.var['idx'][var]:sos.var['idx'][var+1]])
            V = coeff.T * Matrix(ZZSym)
        else:
            dummy = sos.symdecvartable[sos.var['idx'][var]:sos.var['idx'][var+1]]
            V = Matrix(Zsym).T * Matrix(int(np.sqrt(len(dummy))),int(np.sqrt(len(dummy))),dummy).T * Matrix(Zsym)   # Python reshape and MATLAB reshape is different
            V = V[0,0]
        
    elif Type == 'poly':
        temp_MA = sos.symdecvartable[sos.var['idx'][var]:sos.var['idx'][var+1]]
        V = Matrix(temp_MA).dot(Matrix(Zsym))

    return sos, V




    
    

    