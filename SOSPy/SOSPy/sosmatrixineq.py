from .sosineq import sosineq
from sympy import symbols, Matrix
from .sosprogram import sosprogram
from sympy import MatrixBase

def sosmatrixineq(sos:sosprogram, fM:MatrixBase, option:str='quadraticMineq') -> sosprogram:
    '''
    SOSMATRIXINEQ --- Creates a SOS constraint from a matrix inequality
    constraint

    SOSP = sosmatrixineq(SOSP,fM)

    SOSP is the sum of squares program.
    fM is a polynomial matrix used to generate the polynomial inequality y'fMy>0

    option default is quadraticMineq, it asks for sos expression v'M(x)v
    '''

    [n,m] = fM.shape

    if n != m:
        raise ValueError('Matrix fM in inequality fM>0 must be square.')
    
    if option == 'quadraticMineq':
        # creates the vector of variables Mvar to generate the quadratic expression M_var'*fM*Mvar
        if n > sos.varmat['count']:
            Mvar = list(symbols(f'Mvar_:{n}'))
            Mvar = Mvar[sos.varmat['count']:]
            sos.varmat['symvartable'] = sos.varmat['symvartable'] + Mvar    # updates the vartable in the sos program
            sos.varmat['vartable'] = str(sos.varmat['symvartable'])
            sos.varmat['count'] = n
        
        varMconst = sos.varmat['symvartable'][:n]

        symexpr = Matrix([varMconst]) @ fM @ Matrix([varMconst]).transpose()
        sos = sosineq(sos,symexpr,'sparsemultipartite',[sos.symvartable,varMconst])

    elif option == 'Mineq':
        sos = sosineq(sos,fM)

    return sos



