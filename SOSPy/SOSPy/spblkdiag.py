from scipy.sparse import csr_matrix, vstack, hstack, issparse, dia_matrix, coo_matrix

def spblkdiag(A1:dia_matrix, A2:dia_matrix) -> coo_matrix:
    '''
    SPBLKDIAG  Sparse block diagonal concatenation.

        A = spblkdiag(A1,A2)  

    Given two sparse matrices A1 and A2, this function
    constructs their block diagonal concatenation

        | A1 0  |
    A = |       |
        | 0  A2 |

    where A is a sparse matrix.
    '''

    A = vstack([hstack([A1, csr_matrix((A1.shape[0], A2.shape[1]))]), hstack([csr_matrix((A2.shape[0], A1.shape[1])), A2])])

    if not issparse(A):
        raise Exception('A is not sparse')
    
    return A