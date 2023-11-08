from sympy import expand, Matrix
from .sosprogram import sosprogram
from .sosineq import sosineq
from .sossolve import sossolve
import numpy as np
import math
from scipy.linalg import sqrtm
from .proj3 import proj3


def findsos(P,flag='abcdefgh',options=None,verbose=1):
    '''
    FINDMATRIXSOS --- Find a sum of squares decomposition of a given matrix polynomial.

    [Q,Z,decomp,Den] = findsos(P,flag,options)

    P is a symmetric polynomial matrix.

    FLAG is an optional argument which, if set to 'rational', returns a
    sum of squares decomposition with rational coefficients. 

    OPTIONS is an optional argument which specifies the solver and the
    solver-specific parameters on its fields as
    options.solver
    options.params
    the default value for options.solver is 'Mosek'. The solver parameters
    are fields of options.params as, for instance, options.params.tol = 1e-9.

    A positive semidefinite Q and a symbolic monomial vector Z will be
    computed such that

    (Ir kron Z)' * Q * (Ir kron Z) = P(x)

    If P is not a sum of squares, the function will return empty Q and Z.

    If P is a polynomial with integer coefficients and is represented as a
    symbolic object, then [Q,Z] = findsos(P,'rational') will compute a rational
    matrix Q such that Z'*Q*Z = P.
    '''

    if isinstance(P,Matrix):
        dimp = P.shape
        if P.shape[0] != P.shape[1]:
            raise ValueError('The polynomial matrix is not square, it cannot be a sum of squares.')
        if not P.is_symmetric():
            raise ValueError('The polynomial matrix is not symmetric, it can not be a sum of squares.')
    else:
        dimp = [1,1]
    

    P = expand(P)   # Do we need this?

    vartable = list(P.free_symbols)

    # Initialize the SOS program
    prog = sosprogram(vartable)

    # Add the constraint
    prog = sosineq(prog, P)

    # Solve the SOS program
    if options is None:
        options = {}
        options['solver'] = 'cvxpy'
        # params = {}
        # params['tol'] = 1e-9
        # params['alg'] = 2
        # options['params'] = params
    
    prog = sossolve(prog, options,verbose)
    info = prog.solinfo['info']
    
    if 'solver' in options and options['solver'] == 'SDPNAL':
        print('findsos function currently not supported for SDPNAL solver')
        return [],[],[],[]
    if options['solver'] == 'mosek':
        if info['dinf']==1 or info['pinf']==1:
            print('No sum of squares decomposition is found.')
            return [],[],[],[]
    if options['solver'] == 'cvxpy':
        if info['status'] != 'optimal':
            print('No sum of squares decomposition is found.')
            return [],[],[],[]
    
    len_RRx = len(prog.solinfo['RRx'])
    Q = np.array(prog.solinfo['RRx']).reshape(int(math.sqrt(len_RRx)), int(math.sqrt(len_RRx)))
    Z = []
    Z_temp = prog.extravar['Z'][0].toarray()
    for row in Z_temp:
        Z.append(np.prod([var**deg for var, deg in zip(vartable, row)]))
    
    if flag == 'rational':
        A = prog.expr['At'][0].toarray().T
        B = prog.expr['b'][0].toarray()

        # Round x0
        N = 1   # This is the tolerance in the rounding
        xx = prog.solinfo['x']

        kmax = 10
        pd = 0
        while kmax != 0:
            kmax = kmax - 1
            x0 = np.round(N*xx).reshape(-1,1)
            try:
                Qflat,NN = proj3(x0,A,B,N)
                n = math.sqrt(len(Qflat))
                Qr = Qflat.reshape(int(n),int(n)).T
                # Qr should be PSD
                if np.min(np.linalg.eig(Qr/NN)[0]) > -1e-14:
                    kmax = 0
                    pd = 1
                # Increase N, and try again
                N = 2*N
            except:
                NN = 1
                n = math.sqrt(len(xx))
                Qr = xx.reshape(n,n).T
                break
        
        # Experimental, no good error checking yet, so we check that expand(NN*P - Z.'*Qr*Z) is zero!
        try:
            check = expand(NN*P - Matrix(Z).transpose() @ Matrix(Qr) @ Matrix(Z))
            if check != 0 or pd == 0:
                Qr = []
                Z = []
                NN = []
                raise Exception('Could not compute a rational SOS!')
        except:
            Qr = []
            Z = []
            NN = []
            raise Exception('Could not compute a rational SOS!')
        
        Q = Qr
        Den = NN

    else:
        Den = 1

    L = np.real(sqrtm(Q))
    decomp = L @ np.kron(np.eye(dimp[0]), np.array(Z).reshape(-1,1))

    return Q, Z, decomp, Den

        
    



