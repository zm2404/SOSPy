import sympy
from sympy import diff, oo
from .sosconstr import sosconstr


def preprocess(symexpr, var, info):
    # Get the maximum degree of the independent variable
    maxdeg = 0
    dummy = diff(symexpr, var)
    while dummy != 0:
        maxdeg += 1
        dummy = diff(dummy, var)

    # Substitute var
    if info[1] == oo:
        newvar = var + info[0]
        newsymexpr = symexpr.subs(var, newvar)
    elif info[0] == -oo:
        newvar = -var + info[1]
        newsymexpr = symexpr.subs(var, newvar)
    else:
        newvar = (info[1] - info[0]) / 2 * (1 - var) / (1 + var) + (info[1] + info[0]) / 2
        newsymexpr = symexpr.subs(var, newvar) * (1 + var) ** maxdeg

    return newsymexpr



def sosineq(sos, symexpr, info1=None, info2=None):
    '''
    SOSINEQ   Adds an inequality constraint to the SOS program.
    SOSP = sosineq(SOSP,EXPR)
    
    SOSP is the sum of squares program.
    The new constraint is described by EXPR as follows:
    EXPR is a symbolic expression in the variables of the program.
    '''

    # Check if symexpr is a Matrix
    scal_expr = 0
    if not isinstance(symexpr,sympy.MatrixBase):
        scal_expr = 1

    # Interval
    if info1 is not None and not isinstance(info1, str):
        if len(sos.symvartable) != 1:
            raise ValueError('Not a univariate polynomial.')
        if scal_expr != 1:
            for i in range(symexpr.shape[0]):
                #print(symexpr[i])   # 8/21: Why only one index? A vector?
                symexpr[i] = preprocess(symexpr[i], sos.symvartable[0], info1) ## It seems symexpr is a vector?
                sos = sosconstr(sos, 'ineq', symexpr[i])
                sos.expr['type'][sos.expr['num']-1] = 'posint'  # overwrite the type
        else:
            symexpr = preprocess(symexpr, sos.symvartable[0], info1)
            sos = sosconstr(sos, 'ineq', symexpr)
            sos.expr['type'][sos.expr['num']-1] = 'posint'  # overwrite the type

        return sos
    
    # Sparse polynomial
    if info1 == 'sparse':
        if scal_expr != 1:
            for i in range(symexpr.shape[0]):
                sos = sosconstr(sos, 'ineq', symexpr[i])
                sos.expr['type'][sos.expr['num']-1] = 'sparse'  # overwrite the type
        else:
            sos = sosconstr(sos, 'ineq', symexpr)
            sos.expr['type'][sos.expr['num']-1] = 'sparse'  # overwrite the type

        return sos
    

    # Sparse polynomial multipartite
    if info1 == 'sparsemultipartite':
        if scal_expr != 1:
            for i in range(symexpr.shape[0]):
                sos = sosconstr(sos, 'ineq', symexpr[i])
                sos.expr['type'][sos.expr['num']-1] = 'sparsemultipartite'
                sos.expr['multipart'][sos.expr['num']-1] = info2  # 8/18 Edited: is it overwriting? like the above?
        else:
            sos = sosconstr(sos, 'ineq', symexpr)
            sos.expr['type'][sos.expr['num']-1] = 'sparsemultipartite'
            sos.expr['multipart'][sos.expr['num']-1] = info2  # 8/18 Edited: is it overwriting? like the above?

        return sos


    # If info1 and infor2 are empty, then the inequality constraint is
    sos = sosconstr(sos, 'ineq', symexpr)
    return sos