from scipy.sparse import csr_matrix, vstack, hstack, eye, issparse
import numpy as np
import pandas as pd

def findcommonZ(Z1:csr_matrix, Z2:csr_matrix) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    '''
    FINDCOMMONZ --- Find common(distinct) Z and permutation matrices R1, R2

    R1,R2,Z = findcommonZ(Z1,Z2)

    Given two vectors of monomials Z1 and Z2, this 
    function will compute another vector of monomials Z
    containing all the  distinct monomials of Z1 and Z2, and
    permutation matrices R1, R2 such that

    Z1 = R1*Z
    Z2 = R2*Z

    Assumption: all the monomials in Z1, as well as
    the monomials in Z2, are DISTINCT --- but Z1 and Z2 may 
    have common monomials.
    '''

    # Check if Z1 and Z2 are sparse matrix
    if not issparse(Z1):
        Z1 = csr_matrix(Z1)
    if not issparse(Z2):
        Z2 = csr_matrix(Z2)

    if (Z1.shape[0] + Z2.shape[0]) <= 1:
        Z = vstack([Z1, Z2])
        R1 = eye(Z1.shape[0], Z.shape[0])
        R2 = eye(Z2.shape[0], Z.shape[0])

        return R1, R2, Z
        
    # Constructing index matrix
    sizeZ1 = Z1.shape[0]
    Ind1 = np.arange(sizeZ1)[:, None]
    sizeZ2 = Z2.shape[0]
    Ind2 = np.arange(sizeZ2)[:, None]
    Ind = np.block([[Ind1, np.full(Ind1.shape, sizeZ2)], [np.full(Ind2.shape, sizeZ1), Ind2]])
    
    # Constructing Z
    ZZ = vstack([Z1, Z2])
    ZZ_temp = pd.DataFrame(ZZ.toarray())
    IndSort = ZZ_temp.sort_values(by=list(ZZ_temp.columns)).index
    ZZ = ZZ[IndSort]
    ZTemp = np.diff(ZZ.toarray(), prepend=ZZ[-1:].toarray(), axis=0)    # Functionally equivalent to MATLAB code
    I = np.where(np.any(ZTemp != 0, axis=1))[0]
    INull = np.where(np.all(ZTemp == 0, axis=1))[0]
    if I.size == 0:
        I = 0
        INull = 1
    Z = ZZ[I]

    # Constructing permutation matrix
    Ind = Ind[IndSort]
    for i in INull:
        Ind[i - 1, 1] = Ind[i, 1]
        Ind[i, 1] = sizeZ2
    Ind = Ind[I]

    # hstack in scipy.sparse
    R1 = hstack([eye(sizeZ1), csr_matrix((sizeZ1, len(I) - sizeZ1))]).tocsr()
    R1 = R1[:, Ind[:, 0]]
    R2 = hstack([eye(sizeZ2), csr_matrix((sizeZ2, len(I) - sizeZ2))]).tocsr()
    R2 = R2[:, Ind[:, 1]]

    Z = csr_matrix(Z)

    return R1, R2, Z