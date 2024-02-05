from sympy import Matrix, MatrixBase
from .sosconstr import sosconstr
from .sosprogram import sosprogram
from sympy.core.add import Add

def soseq(sos:sosprogram, symexpr:MatrixBase|Add) -> sosprogram:
    '''
    sos = soseq(sos,symexpr)
    SOSEQ --- Add a new equality constraint f(x) = 0
    to an SOS program 

    OUTPUTS:
    SOSP: The modified SOS program structure.

    INPUTS:
    SOSP: The SOS program structure to be modified
    SYMEXPR: The expression on the left hand side of the constraint, i.e., f(x).
    This expression may be scalar, matrix, or vector-valued. 
    '''

    # Check if symexpr is a Matrix
    if not isinstance(symexpr, Matrix):
        sos = sosconstr(sos, 'eq', symexpr)
    else:
        dimp = symexpr.shape[0]
        for i in range(dimp):
            sos = sosconstr(sos, 'eq', symexpr[i])

    return sos
    
        