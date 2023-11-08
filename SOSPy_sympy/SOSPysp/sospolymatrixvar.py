from sympy import Matrix
from .sospolyvar import sospolyvar

def sospolymatrixvar(sos,ZSym,n,matrixstr=None):
    '''
    SOSPOLYMATRIXVAR --- Declare a polynomial matrix variable P in the sos program
    SOS of

    SOSP,P = sospolymatrixvar(SOSP,ZSym,n,matrixstr)

    OUTPUTS:
    SOS: the modified sum of squares program structure.
    P: the new polynomial matrix variable. If SOS contains symbolic variables, the
    output type of P will be sym. Otherwise, the output type defaults to
    dpvar unless the caller specifies output='pvar' as a fifth input.

    INPUTS
    SOS: The sum of squares program structure to be modified.
    ZSym[list]: The list of monomials to be contained in P. Decision
    variables corresponding to those monomials will be assigned
    automatically by SOSPOLYVAR.
    n: The desired dimension of the output matrix P: n(1) x n(2)
    matrixstr (optional): a char string with the option 'symmetric' when required
    '''

    if matrixstr is not None:
        if matrixstr == 'symmetric':
            if n[0] == n[1]:
                P = Matrix.zeros(n[0])
                for i in range(n[0]):
                    for j in range(n[0]):
                        sos,var = sospolyvar(sos,ZSym)
                        P[i,j] = var
                        P[j,i] = var

                return sos, P
            # omit two other cases in MATLAB code
            else:
                print("'symmetric' option used, matrix must be square.")
                P = []
                return sos, P
        else:
            print("matrix structure 'matrixstr' is not defined.")
            P = []
            return sos, P
        
    else:
        
        P = Matrix.zeros(n[0],n[1])
        for i in range(n[0]):
            for j in range(n[1]):
                sos,var = sospolyvar(sos,ZSym)
                P[i,j] = var
        if n[0] == n[1] == 1:
            P = P[0,0]

        return sos, P


        