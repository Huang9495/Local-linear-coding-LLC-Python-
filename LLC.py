import numpy as np
import numpy.linalg as lg
import copy
import cv2

def getLLCResut(A, B, knn, beta):

#############################################################
# find k nearest neighbors
#       B       -M x d codebook, M entries in a d-dim space
#       X       -N x d matrix, N data points in a d-dim space
#---------------------
###############################################################

    A = np.mat(A,dtype=np.float32)
    B = np.mat(B,dtype=np.float32)
    nframe = B.shape[0]
    nbase = A.shape[0]
    dims = A.shape[1]
    alphas = np.zeros((nframe, nbase))

    if A.shape[1] == B.shape[1]:
        XX = np.sum(np.multiply(B,B),axis=1)
        AA = np.sum(np.multiply(A,A),axis=1)
        AAT = np.transpose(AA)
        AT = np.transpose(A)
        a = np.repeat(XX, nbase, axis=1)
        b = 2 * B * AT
        c = np.repeat(AAT, nframe, axis=0)
        D = a - b + c 
        IDX = np.zeros((nframe, knn),dtype=np.int)
        
        for i in range(0,nframe):
            d = copy.deepcopy(D[i])
            idx1 = np.array(copy.deepcopy(np.sort(d,axis=1)))
            IDX[i][:] = copy.deepcopy(idx1[i][0:knn])

        II = np.eye(knn, knn)
        for i in range(1,nframe):
            idx = IDX[i][:];
            A_new = list(range(0,len(idx)))
            for j in range(0,len(idx)):
                A = np.array(A)
                A_new[j] = copy.deepcopy(A[idx[j]])
                
            z =  A_new - np.repeat(B[i][:], knn, axis=0)
            zt = np.transpose(z)
            C = z * zt
            C = C + II * beta * np.trace(C)        
            C = np.mat(C)
            C_inv = lg.inv(C)
            w = C_inv * np.ones((knn,1),dtype=int)
            w = w / np.sum(w,axis=0)
            wt = np.array(np.transpose(w))
            for z in range(0,len(idx)):
                alphas[i][idx[z]] = wt[0][z]
    return alphas
