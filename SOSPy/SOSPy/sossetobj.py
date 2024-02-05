from sympy import diff, simplify
from .sosprogram import sosprogram

def sossetobj(sos:sosprogram, symexpr) -> sosprogram:
    '''
    SOSSETOBJ --- Set the objective function of an SOS program. 

    SOSP = sossetobj(SOSP,EXPR)

    Given a sum of squares program SOSP and an expression EXPR, SOSSETOBJ
    makes EXPR the objective function of the sum of squares program SOSP, 
    i.e., EXPR will be minimized.
    '''

    objvartable_temp = symexpr.free_symbols
    decvartable = sos.symdecvartable

    idx = []
    for i, sym_var in enumerate(objvartable_temp):
        if sym_var not in decvartable:
            raise ValueError('There is unknown decision variable')
        idx.append(decvartable.index(sym_var))
        tempexpr = diff(symexpr, sos.symdecvartable[idx[-1]])
        if len(tempexpr.free_symbols) != 0:
            raise ValueError('Not a linear objective function.')
        else:
            sos.objective = sos.objective.tolil()   # 8/18 added
            sos.objective[idx[-1]] = float(tempexpr)
            sos.objective = sos.objective.tocsr()   # 8/18 added
        symexpr = simplify(symexpr - float(tempexpr)*sos.symdecvartable[idx[-1]])

    return sos
        