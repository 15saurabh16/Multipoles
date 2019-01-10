# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:15:06 2019

@author: Saurabh
"""

import scipy.stats.mstats as statm
import os
import numpy as np
import sys
import numpy.linalg as LA
import time
import pdb


def generate_std_rand_ts_corrmat(CorrMat,T):
    
    N = int(CorrMat.shape[0])
    S = np.zeros((T,N))
    
    for i in range(N):
        S[:,i]= np.random.normal(loc=0.0,scale=1.0,size=T)
   
    L = LA.cholesky(CorrMat)
    
    X = np.matmul(S,L)
    X = statm.zscore(X,axis=0)
    return X

    
    
#     CorrMat[0,1] = CorrMat[0,2] = CorrMat[1,0] = CorrMat[1,2] = CorrMat[2,0] = CorrMat[2,1] = -0.2