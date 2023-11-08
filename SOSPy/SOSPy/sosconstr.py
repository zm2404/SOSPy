from .getequation import getequation

def sosconstr(sos,Type,symexpr):
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