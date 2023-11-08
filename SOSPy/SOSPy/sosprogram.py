import numpy as np
from scipy.sparse import lil_matrix
import importlib
import cvxpy as cp

class sosprogram:
    def __init__(self, vartable=[], decvartable=[]):
        # Check if SDP solvers are installed
        
        sdp_solvers = ['cvxpy', 'mosek']    # so far only cvxpy and mosek are supported, but cvxpy can call other solvers
        if not any(importlib.util.find_spec(solver) is not None for solver in sdp_solvers):
            raise ImportError('No SDP solvers found.')
        
        SOLVERS = ['MOSEK', 'CVXOPT', 'SCS', 'COPT', 'SDPA']
        INSTALLED_SOLVERS = cp.installed_solvers()
        SOLVER = [item for item in SOLVERS if item in INSTALLED_SOLVERS]
        if len(SOLVER) == 0:
            raise ImportError('No SDP solvers found.')
        print('Installed SDP solvers: ', SOLVER)
        
        # self.var contains information about variables created by initial variables.
        # Z is the matrix of monomial exponents, we can get monomials from Z and vartable.
        # Polynomial or SOS polynomial are indicated by 'type', and their monomials are indicated by 'Z'.
        self.var = {
            'num' : 0,
            'type': {},
            'Z': {},    # monomial degree matrix whose entries are the monomial exponents.
            'ZZ': {},   
            'T': {},
            'idx': {}   # list needs to be initialized, but dict doesn't need to be initialized.
        }

        # self.expr contains information about user's constraints.
        self.expr = {
            'num': 0,
            'type': {},
            'At': {},   # transpose of the coefficient matrix of the constraint
            'b': {},    # right hand side of the constraint
            'Z': {},    # monomial exponents of the constraint
            'multipart': {} # whether the constraint is a multipartite SOS constraint
        }

        # Filled in sossolve(), create decision variables for SDP solver
        self.extravar = {
            'num': 0,
            'Z': {},
            'ZZ': {},
            'T': {},
            'idx': {}
        }

        # weights of the objective function
        self.objective = []     

        # self.solinfo contains information about the SDP solution of the problem.
        self.solinfo = {
            'x': [], 
            'y': [], 
            'RRx': [], 
            'RRy': [], 
            'info': {},
            'solverOptions': {},
            'var' : {},
            'extravar': {},
            'decvar': {}
        }

        # np.shape() reuturns (n,1) for a column vector, but returns (n,) for a row vector.
        # So len(np.shape(vartable)) > 1 indicates that vartable is a column vector.
        # We need to transpose it to a row vector.
        # if vartable is not None and len(np.shape(vartable)) > 1:
        #    vartable = np.transpose(vartable)
        # 7/2 edit: use flatten() to convert whatever shape to a row vector
        if isinstance(vartable, list):
            vartable = np.array(vartable)   # doesn't need flatten()
        else:
            vartable = vartable.flatten()

        if isinstance(decvartable, list):
            decvartable = np.array(decvartable)
        else:
            decvartable = decvartable.flatten()

        self.symvartable = vartable.tolist()        # self.symvartable is a sympy matrix
        self.vartable = str(self.symvartable)       # self.vartable is a string

        # self.varmat contains information about matrix constraints.
        self.varmat = {
            'vartable': '[]', 
            'symvartable': [], 
            'count': 0
        }

        if len(decvartable) != 0:
            self.objective = lil_matrix((len(decvartable), 1))
            self.symdecvartable = decvartable.tolist()      # self.symdecvartable contains decision variables but in a sympy matrix
            self.decvartable = str(self.symdecvartable)     # self.decvartable is a string
            self.var['idx'][0] = len([i for i in self.decvartable if i==',']) + 1   
        else:
            self.decvartable = '[]'
            self.symdecvartable = []
            self.var['idx'][0] = 0