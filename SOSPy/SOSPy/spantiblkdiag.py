from scipy.sparse import csr_matrix, vstack, hstack, issparse


def spantiblkdiag(A1,A2):
    '''
    SPANTIBLKDIAG  Sparse anti block diagonal concatenation.

        A = spantiblkdiag(A1,A2)  

    Given two sparse matrices A1 and A2, this function
    constructs their anti block diagonal concatenation

        | 0  A1 |
    A = |       |
        | A2  0 |

    where A is a sparse matrix.
    '''

    A = vstack([hstack([csr_matrix((A1.shape[0],A2.shape[1])),A1]),hstack([A2,csr_matrix((A2.shape[0],A1.shape[1]))])])

    if not issparse(A):
        raise Exception('A is not sparse')
    
    return A
