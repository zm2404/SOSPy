import numpy as np
import pandas as pd
from sympy import expand, collect, poly, lambdify, Matrix, together, cancel, Add
from scipy.sparse import csr_matrix, lil_matrix
from sympy.tensor.array import derive_by_array
import sympy

def sortNoRepeat(Z, Zin):
    if Zin.size == 0:
        return Z
    if Z.size == 0:
        return pd.DataFrame(Zin).sort_values(by=list(range(Zin.shape[1])),ascending=False).values
    
    Z = pd.DataFrame(Z)
    Zin = pd.DataFrame(Zin)
    Znew = pd.concat([Z, Zin]).drop_duplicates().values

    Znew = pd.DataFrame(Znew).sort_values(by=list(range(Znew.shape[1])),ascending=False).values

    return Znew


def approx_zero(expr, tolerance=1e-10):
    new_terms = []
    for term in expr.as_ordered_terms():
        coefficient, _ = term.as_coeff_Mul()
        if abs(coefficient) > tolerance:
            new_terms.append(term)
    return Add(*new_terms)


def poly_approx_zero(polynomial):
    '''
    Round small coefficients to zero in a polynomial.
    '''
    numerator, denominator = polynomial.as_numer_denom()
    new_numerator = approx_zero(numerator)
    new_denominator = approx_zero(denominator)

    new_polynomial = new_numerator / new_denominator

    return new_polynomial


def getequation(expr, symvartable, decvartable, varmat):
    '''
    function At,b,Z = getequation(expr,symvartable,decvartable,decvartablename)

    GETEQUATION --- Convert a symbolic expression to At, b, and Z
            used in an SOS program. In this format, 
    expr = Im kron [1][ C ]In kron Z
                   [x][   ]                                                                
         = (b+x^TAt^T) Imn kron Z
    Inputs:
    expr: This expression is in symbolic format. 
    It may be matrix or scalar valued. At present, matrix-valued inputs should 
    only be used with equality constraints
    symvartable: List of the independent variables in the sosprogram
    '''
    
    if not decvartable:  # checks if decvartable is an empty list
        decvarnum = 0
    else:
        decvarnum = len(decvartable)

    if len(str(varmat))-2 == 2:
        vartable = symvartable
    else:
        vartable  = symvartable + varmat  # If they are sympy, + means concatenate.


    # If expr is a scalar, then FPexpr.shape is an empty tuple.
    # Thus, we need to check if expr is a Matrix.
    if isinstance(expr, sympy.MatrixBase):
        FPexpr = Matrix(expand(expr))
        dimp = FPexpr.shape[1]

        for i in range(FPexpr.shape[0]):
            for j in range(dimp):
                FPexpr[i,j] = collect(FPexpr[i,j], vartable)
                FPexpr[i,j] = together(FPexpr[i,j])         # Following are used to gaurantee the accuracy of the coefficients.
                FPexpr[i,j] = cancel(FPexpr[i,j])
                FPexpr[i,j] = poly_approx_zero(FPexpr[i,j])
                FPexpr[i,j] = cancel(FPexpr[i,j])

        # if decvartable is not empty.
        if decvartable:
            Zfull = np.array([])
            for i in range(dimp):
                for j in range(i, dimp):
                    p = poly(FPexpr[i, j], vartable)
                    coefmonmatr = [[*monomial] for monomial in p.monoms()]
                    Z = np.array(coefmonmatr)
                    Zfull = sortNoRepeat(Zfull, Z)

            Z = csr_matrix(Zfull)
            nmon, nvar = Z.shape
            coeffnts = lil_matrix((nmon * dimp, dimp))
            coeffnts_decvar_dict = {} # use this

            for i in range(dimp):
                for j in range(i, dimp):
                    p1 = poly(FPexpr[i, j], vartable)
                    coefmon = list(p1.coeffs())
                    # First I get all coefficients and monomials of the polynomial, 
                    # then I substitute all the decision variables with 0
                    coeff_bef_sub = p1.coeffs()
                    coeff_aft_sub = [cf.subs(dict(zip(decvartable, np.zeros(len(decvartable))))) for cf in coeff_bef_sub]
                    coefmonmatr = [[coeff, *monom] for coeff, monom in zip(coeff_aft_sub, p1.monoms())]

                    for k in range(len(coefmonmatr)):
                        s_ijk = coefmonmatr[k][0]

                        s_ijk_decvar = coefmon[k]

                        mon_k = coefmonmatr[k][1:]
                        ind_k = np.argmax(np.sum((Zfull == np.kron(np.ones((nmon,1)),mon_k)), axis=1))
                        coeffnts[ind_k*dimp+i, j] = s_ijk
                        coeffnts[ind_k*dimp+j, i] = s_ijk

                        # coeffnts_decvar is a sparse matrix so it cannot store Sympy values
                        # we can try to use dictionaly to store the values
                        coeffnts_decvar_dict[(ind_k*dimp+i, j)] = s_ijk_decvar
                        coeffnts_decvar_dict[(ind_k*dimp+j, i)] = s_ijk_decvar


        # if decvartable is empty.
        else: 
            Zfull = np.array([])
            for i in range(dimp):
                for j in range(i, dimp):
                    p = poly(FPexpr[i, j], vartable)
                    coefmon = [[*monomial] for monomial in p.monoms()]
                    Z = np.array(coefmon)
                    Zfull = sortNoRepeat(Zfull, Z)

            Z = csr_matrix(Zfull)
            nmon, nvar = Z.shape
            coeffnts = lil_matrix((nmon * dimp, dimp))

            for i in range(dimp):
                for j in range(i, dimp):
                    p = poly(FPexpr[i, j], vartable)
                    coefmon = [[coeff, *monom] for coeff, monom in zip(p.coeffs(), p.monoms())]

                    for k in range(len(coefmon)):
                        s_ijk = coefmon[k][0]
                        mon_k = coefmon[k][1:]
                        ind_k = np.argmax(np.sum((Zfull == np.kron(np.ones((nmon,1)),mon_k)), axis=1))
                        coeffnts[(ind_k)*dimp+i, j] = s_ijk
                        coeffnts[(ind_k)*dimp+j, i] = s_ijk


        
    # If expr is a scalar, then FPexpr.shape is an empty tuple. we cannot do FPexpr.shape[1].
    else:
        dimp = 1
        FPexpr = expand(expr)
        FPexpr = collect(FPexpr, vartable)
        FPexpr = together(FPexpr)
        FPexpr = cancel(FPexpr)
        FPexpr = poly_approx_zero(FPexpr)
        FPexpr = cancel(FPexpr)


        # if decvartable is not empty.
        if decvartable:
            Zfull = np.array([])
            p = poly(FPexpr, vartable)
            coefmonmatr = [[*monomial] for monomial in p.monoms()]
            Z = np.array(coefmonmatr)
            Zfull = sortNoRepeat(Zfull, Z)

            Z = csr_matrix(Zfull)
            nmon, nvar = Z.shape
            coeffnts = lil_matrix((nmon * dimp, dimp))
            coeffnts_decvar_dict = {} # use dictionary instead

            p1 = poly(FPexpr, vartable)
            coefmon = list(p1.coeffs())
            # First I get all coefficients and monomials of the polynomial, 
            # then I substitute all the decision variables with 0
            coeff_bef_sub = p1.coeffs()
            coeff_aft_sub = [cf.subs(dict(zip(decvartable, np.zeros(len(decvartable))))) for cf in coeff_bef_sub]
            coefmonmatr = [[coeff, *monom] for coeff, monom in zip(coeff_aft_sub, p1.monoms())]

            for k in range(len(coefmonmatr)):
                s_ijk = coefmonmatr[k][0]
                s_ijk_decvar = coefmon[k]

                mon_k = coefmonmatr[k][1:]
                ind_k = np.argmax(np.sum((Zfull == np.kron(np.ones((nmon,1)),mon_k)), axis=1))
                coeffnts[ind_k * dimp, 0] = s_ijk
                coeffnts_decvar_dict[(ind_k*dimp, 0)] = s_ijk_decvar

        # if decvartable is empty.
        else: 
            Zfull = np.array([])
            p = poly(FPexpr, vartable)
            coefmon = [[*monomial] for monomial in p.monoms()]
            Z = np.array(coefmon)
            Zfull = sortNoRepeat(Zfull, Z)

            Z = csr_matrix(Zfull)
            nmon, nvar = Z.shape
            coeffnts = lil_matrix((nmon * dimp, dimp))

            p = poly(FPexpr, vartable)
            coefmon = [[coeff, *monom] for coeff, monom in zip(p.coeffs(), p.monoms())]

            for k in range(len(coefmon)):
                s_ijk = coefmon[k][0]
                mon_k = coefmon[k][1:]
                ind_k = np.argmax(np.sum((Zfull == np.kron(np.ones((nmon,1)),mon_k)), axis=1))
                coeffnts[ind_k * dimp, 0] = s_ijk


    
    At = lil_matrix((decvarnum, Z.shape[0] * dimp ** 2))
    if decvartable:
        for k in range(Z.shape[0]):
            # Extract coefficients of decision variables from the dictionary
            Mivec = np.array([coeffnts_decvar_dict.get((i,j), 0) for i in range(k*dimp, (k+1)*dimp) for j in range(dimp)]).reshape(dimp**2,)
            jac_func = lambdify(decvartable, derive_by_array(Mivec, decvartable))
            # use * to unpack the list so that the function can take multiple arguments.
            At[:, k * dimp**2:(k+1) * dimp**2] = -jac_func(*([0] * len(decvartable)))   

    At = csr_matrix(At)
    b = csr_matrix(coeffnts)
    return At, b, Z