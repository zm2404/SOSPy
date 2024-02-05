from .getequation import getequation
from .sosprogram import sosprogram
from sympy import MatrixBase
from sympy.core.add import Add

def sosconstr(sos:sosprogram,Type:str,symexpr:Add|MatrixBase) -> sosprogram:
    '''
    SOSCONSTR   Adds a constraint to the SOS program.
    SOSP = sosconstr(SOSP,TYPE,EXPR)
    
    SOSP is the sum of squares program.
    The new constraint is described by TYPE as follows:
    '''
    expr = sos.expr['num']     # expr is an index
    sos.expr['num'] = sos.expr['num'] + 1
    sos.expr['type'][expr] = Type  # Update the type of the constraint

    # assume symexpr is a sympy expression, and sos.symvartable, sos.symdecvartable and sos.varmat.symvartable are lists of sympy expressions.
    sos.expr['At'][expr], sos.expr['b'][expr], sos.expr['Z'][expr] = getequation(symexpr, sos.symvartable, sos.symdecvartable, sos.varmat['symvartable'])

    return sos