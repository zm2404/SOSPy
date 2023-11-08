from scipy.sparse import csr_matrix, vstack, eye, issparse, lil_matrix
from .findcommonZ import findcommonZ

def getconstraint(Z):
    '''
    GETCONSTRAINT --- Find constraint for sum of squares decomposition.

    A,ZZ = getconstraint(Z)

    Z is a monomial vector description.
    This function computes the constraint matrix A and the polynomial
    vector ZZ, such that if q satisfies

    A*q = F'

    Then

    Z'*Q*Z = F*ZZ      (where Q is the matrix form of q)
    '''

    # First write Z'*Q*Z as
    # Z'*Q*Z = (e1'*Q*R1 + e2'*Q*R2 + ... + en'*Q*Rn) ZZ

    if not issparse(Z):
        Z = csr_matrix(Z)

    sizeZ = Z.shape[0]
    ZZ = Z + vstack([Z[0]]*sizeZ)
    M = {}  # Let M be a dictionary, M(1).R = M{1} ...
    M[0] = eye(sizeZ)
    for i in range(1,sizeZ):
        Ztemp = Z + vstack([Z[i]]*sizeZ)
        R1,R2,ZZ = findcommonZ(ZZ,Ztemp)
        for j in range(i):
            M[j] = M[j] @ R1
        M[i] = R2

    # Construc the constaint equations
    Q = lil_matrix((sizeZ,sizeZ))
    A = lil_matrix((ZZ.shape[0],sizeZ**2))
    for i in range(sizeZ**2):
        Q[i % sizeZ, i // sizeZ] = 1
        j, k = Q.nonzero()
        A[:,i] = M[j[0]].T @ Q[j[0],:].T
        Q[i % sizeZ, i // sizeZ] = 0

    A = csr_matrix(A)
    return A, ZZ


