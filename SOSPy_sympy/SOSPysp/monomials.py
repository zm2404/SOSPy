import numpy as np
from itertools import combinations_with_replacement
from scipy.sparse import csr_matrix
import sympy



def monomials(vartable,degree):
    '''
    Construct a vector of monomials with prespecified degrees.
    Z = monomials(VARTABLE, DEGREE)

    Given a list of independent variables VARTABLE and a list of non-negative 
    integers DEGREE, this function constructs a column vector 
    containing all possible monomials of degree described in DEGREE.

    For example, monomials([x1,x2],[1,2,3]) with x1 and x2 
    being symbolic variables will return exponent matrix of monomials in x1 and 
    x2 of degree 1, 2, and 3.

    
    Alternative calling convention:

    Z = monomials(n,d)

    Given two integers n and d, this function constructs 
    a vector containing all possible monomials of degree d 
    in n variables.

    n is the number of variables
    d is the degree of monomials
    Z is the exponent matrix of monomial vector
    '''

    # Check if vartable is a single integer, if so consider it as the number of variables
    if isinstance(vartable, int):
        num_variables = vartable
    else:
        num_variables = len(vartable)
    
    Z = []

    for d in degree:
        for comb in combinations_with_replacement(range(num_variables), d):
            z = np.zeros(num_variables, dtype=int)
            for idx in comb:
                z[idx] += 1
            Z.append(z)
    
    if isinstance(vartable, list) and all(isinstance(item, sympy.core.symbol.Symbol) for item in vartable):
        res = []
        for row in Z:
            exprs = [var**deg for var, deg in zip(vartable, row)]
            res.append(sympy.prod(exprs))
        return res
    else:
        return csr_matrix(Z)