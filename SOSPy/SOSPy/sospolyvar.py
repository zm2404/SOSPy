from sympy import symbols, MatrixBase
from .sosvar import sosvar

from .sosprogram import sosprogram
from sympy.core.symbol import Symbol

def sospolyvar(sos:sosprogram, Zsym:int|float|list[Symbol]) -> tuple[sosprogram, MatrixBase]:
    '''
    SOSPOLYVAR --- Declare a new scalar polynomial variable in
        an SOS program

    SOS, V = sospolyvar(sos,ZSym)

    OUTPUTS:
    SOS: the modified sum of squares program structure.
    V: the new polynomial variable.

    INPUTS
    SOS: The sum of squares program structure to be modified.
    ZSym: The vector of monomials to be contained in V. Decision
    variables corresponding to those monomials will be assigned
    automatically by SOSPOLYVAR.
    '''

    if (isinstance(Zsym, int) or isinstance(Zsym, float)) and Zsym == 1:
        Zsym = symbols(Zsym)

    sos, V = sosvar(sos,'poly',Zsym)

    return sos, V
