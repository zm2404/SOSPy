import numpy as np
from fractions import Fraction

def proj3(x0,A,b,N):
    '''
    Orthogonal projection of the point x0 onto the subspace Ax=b 

    Here, all rational data
    A is assumed fat, and full row rank
    A*A' is diagonal

    All integer computations
    The solution is x/NN, the input is x0/N
    '''
    # x0, A, b are all numpy arrays
    # N is an integer
    
    aa = np.sum(A * A, axis=1)
    N2 = np.lcm.reduce(np.unique(aa).astype(int).tolist())
    x = N2*x0 - A.T @ ((N2 / aa).reshape(-1,1) * (A@x0-b*N))

    NN = N2*N

    Den = [[Fraction(val).limit_denominator().denominator for val in row] for row in x]
    x = x * np.prod(np.unique(Den))
    NN = NN * np.prod(np.unique(Den))

    # Make sure it's minimal
    ix = np.unique(abs(x)).nonzero()
    try:
        cfact = np.gcd.reduce(np.unique(abs(x))[ix].astype(int).tolist() + [NN])
    except:
        raise Exception('No reasonable projection with rational values is possible')
        
    x = x / cfact
    NN = NN / cfact

    return x,NN



