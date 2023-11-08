from sympy import sympify
from .sosvar import sosvar

def sossosvar(sos, Zsym):
    '''
    SOSSOSVAR --- Declare a new sum of squares variable in
        an SOS program

    sos, V = sossosvar(SOSP,Z)

    OUTPUTS:
    SOS: the modified sum of squares program structure.
    V: the new sum of squares decision variable. If SOS contains symbolic variables, the
    output type of P will be sym. Otherwise, the output type defaults to
    dpvar unless the caller specifies PVoption='pvar' as a fourth input.

    INPUTS
    SOS: The sum of squares program structure to be modified.
    ZSym[LIST]: the list of monomials contained in the Gram
    decomposition of VAR, i.e.,

        VAR = ZSym' * COEFF * ZSym

    where COEFF is a coefficient matrix that is restricted to be
    positive semidefinite. COEFF and the decision variables contained
    in it will be constructed automatically by SOSSOSVAR.
    '''
    if isinstance(Zsym, int) or Zsym == 1:
        Zsym = sympify(Zsym)

    sos, V = sosvar(sos, 'sos', Zsym)

    return sos, V
    