from .inconvhull import inconvhull
import numpy as np

def sparsemultipart(Z1,Z2,info):
    '''
    Find the elements in Z1 that are in the convex hull of Z2, where Z2 is bipartite
    '''
    
    I = [i for i,x in enumerate(Z2.sum(axis=0).tolist()[0]) if x==0]
    sizeinf = len(info)

    if sizeinf == 1:
        raise Exception('Error in sparsemultipart option - at least two sets are required')
    Z3 = []
    for i in range(sizeinf):
        Z3_temp = inconvhull(Z1[:,info[i]],Z2[:,info[i]])
        Z3.append(Z3_temp)
        
    for i in range(sizeinf-1):
        Z3[i+1] = np.hstack([np.vstack([Z3[i]]*Z3[i+1].shape[0]),np.kron(Z3[i+1],np.ones((Z3[i].shape[0],1)))])
        
    Z3 = Z3[-1]
    Z = np.zeros((Z3.shape[0],Z2.shape[1]))
    lgth = 0

    for i in range(sizeinf):
        Z[:,info[i]] = Z3[:,lgth:(lgth+len(info[i]))]
        lgth = len(info[i])+lgth

    Z[:,I] = np.zeros((Z.shape[0],len(I)))

    return Z