import numpy as np
from scipy.linalg import orth
from scipy.spatial import ConvexHull
from scipy.linalg import null_space
import sympy


def useconvhulln(Z2):
    convh = ConvexHull(Z2).simplices
    facets = convh.shape[0]
    nZ1,nZ2 = Z2.shape

    coeff = []
    cold = []
    for i in range(facets):
        vert = np.hstack([Z2[convh[i]],1*np.ones((nZ2,1))])
        M = sympy.Matrix(vert)
        nullvert = np.array(M.nullspace()[0])
        if nullvert.shape[1] == 1:
            if nullvert[-1] != 0:
                nullvert = nullvert / nullvert[-1]
            coeff.append(nullvert[:-1].flatten().tolist())
            cold.extend(nullvert[-1].flatten().tolist())
    coeff = np.array(coeff,dtype='float')
    cold = np.array(cold,dtype='float').reshape(-1,1)

    # Condition the matrix a bit
    coeffcold = np.hstack([coeff,cold])

    # Discard same hyperplanes (due to convhulln result)
    coeff2cold, Ix = np.unique(coeffcold, axis=0, return_index=True)

    # Remove a possible zero row
    yind = [i for i,x in enumerate(np.sum(coeff2cold**2,axis=1)) if x==0]
    if yind:
        Ix = np.hstack([Ix[:yind[0]],Ix[yind[0]+1:]])
        
    coeff2 = coeff[Ix]
    convhnew = convh[Ix]
    cnew = cold[Ix]
    facetsnew = convhnew.shape[0]

    # Make inequalities out of them by testing a point not on the hyperplane
    # Notation: convex hull is now Ax-b<=0
    for fac in range(facetsnew):
        for ind in range(nZ1):
            matr = [i for i,x in enumerate(convhnew[fac]-1) if x==0]
            tests = -1*coeff2[fac] @ Z2[ind].T - cnew[fac]
            if (not matr) and np.abs(tests) > 1e-8:
                break
        if tests > 0:
            coeff2[fac] = -1*coeff2[fac]
            cnew[fac] = -1*cnew[fac]
            
    A = coeff2
    B = cnew

    return A, B



def inconvhull(Z1, Z2):
    # First, find the affine subspace where everything lives
    # (for instance, in the homogeneous case)

    # Translate so it goes throught the origin
    mr = np.mean(Z2, axis=0)
    Rzero = np.array(Z2 - mr)

    # The columns of N generate the subspace
    N = null_space(Rzero)

    # Z2*N should be constant
    cval = np.mean(Z2 @ N, axis=0)

    # Get only the monomials in the subspace
    tol = 0.01
    ix = np.where(np.sum(np.abs(Z1 @ N - cval), axis=1) < tol)
    nZ1 = Z1[ix]

    # Now, the inequalities:
    # Find an orthonormal basis for the subspace
    # (I really should do both things at the same time...)

    # Project to the lower dimensional space, so convhull works nicely
    Q = orth(Rzero.T)

    # This calls CDD, or whatever
    if (Z2 @ Q).shape[1] > 1:
        A, B = useconvhulln(Z2 @ Q)

        # Find the ones tht satisfy the inequalities, and keep them.
        ix = np.where(np.min(B + A@Q.T@Z1[ix].T,axis=0)>-tol)[0]
        Z3 = nZ1[ix]

    elif (Z2 @ Q).shape[1] == 1:
        A = np.array([1,-1]).reshape(-1,1)
        B = np.array([np.max(Z2 @ Q), -np.min(Z2 @ Q)]).reshape(-1,1)
        ix = np.where(np.min(B - A @ Q.T @ nZ1.T, axis=0) > -tol)
        Z3 = nZ1[ix]
    else:
        Z3 = nZ1

    Z3 = np.unique(Z3.toarray(), axis=0)

    return Z3

