from IPython.display import display, Math
import sympy

from .sosprogram import sosprogram
from sympy.core.symbol import Symbol
from sympy.core.add import Add

def mysymsubs(expr:Symbol|Add, old:list[Symbol], new:list[float], digit:int) -> float|Add:
    '''
    MYSYMSUBS --- Fast symbolic substitution.

    NEWEXPR = mysymsubs(EXPR,OLD,NEW,DIGIT)

    EXPR is symbolic/string.
    OLD is symbolic.
    NEW is numeric (column vector).
    DIGIT is the number of digits.
    '''
    new = [round(item, digit) for item in new]
    replacements = {old[i]: new[i] for i in range(len(old))}
    new_expr = expr.subs(replacements)
    return new_expr


def sosgetsol(sos:sosprogram, V, solname:str=None, digit:int=5) -> float|Add:
    '''
    SOSGETSOL --- Get the solution from a solved SOS program 

    SOL = sosgetsol(SOSP,VAR,DIGIT) 

    SOL is the solution from a solved sum of squares program SOSP,
    obtained through substituting all the decision variables
    in VAR by the numerical values which are the solutions to
    the corresponding semidefinite program. 

    The third argument DIGIT (optional) will determine the 
    accuracy of SOL in terms of the number of digits. Default 
    value is 5.
    '''
    p = mysymsubs(V,sos.symdecvartable,sos.solinfo['RRx'][:len(sos.symdecvartable)],digit)

    latex_code = sympy.latex(p)
    if solname is None:
        solname = str(V)
    display(Math(solname+"= "+latex_code))
    return p