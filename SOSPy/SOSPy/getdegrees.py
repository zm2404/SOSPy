import numpy as np
from sympy import poly

def getdegrees(Z, symvartable):
    '''
    GETDEGREES --- Get degrees of monomials.

    ZDEG = getdegrees(Z,VARTABLE)

    Z is a list of monomials.
    VARTABLE is the list of independent variables.
    ZDEG is the degree of Z (in SOSTOOLS notation)
    '''

    p = np.sum(Z)
    p = poly(p, symvartable)
    coefmonmatr = [[*monomial] for monomial in p.monoms()]
    Zdeg = np.array(coefmonmatr)

    return  Zdeg