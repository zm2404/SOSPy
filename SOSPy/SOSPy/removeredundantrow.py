import scipy
import numpy as np

def remove_redundant_row(A:np.ndarray,b:np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ''' 
    Remove redundant rows from the matrix A, and do the same to the vector b.
    
    Input: A is a numpy array, b is a numpy array.
    Output: A is a numpy array, b is a numpy array, rows_to_keep is a list of indices of rows to keep.
    '''

    TOL = 1e-10
    Q, R, P = scipy.linalg.qr(A, pivoting=True)  # pivoting helps robustness
    rows_to_keep = []
    for i in range(R.shape[0]):
        if np.linalg.norm(R[i, :]) > TOL:
            rows_to_keep.append(i)
    R = R[rows_to_keep, :]
    Q = Q[:, rows_to_keep]
    # Rearrange P.
    Pinv = np.zeros(P.size, dtype='int')
    for i in range(P.size):
        Pinv[P[i]] = i
    # Rearrage R.
    R = R[:, Pinv]
    A = R
    b_old = b
    b = Q.T.dot(b)  # inv(Q)*b = (QT*Q)^-1 * QT * b = I^-1 * QT * b = QT * b
    # If b is not in the range of Q, the problem is infeasible.
    if not np.allclose(b_old, Q.dot(b)):
        return None, None, None
    
    return A, b, rows_to_keep
    

