# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:23:07 2018

@author: agraw066
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:41:30 2017

@author: agraw066
"""

# import BUD_multipoles_parallel as 

import scipy.stats.mstats as statm
import os
import numpy as np
import scipy.io as sio
import sys
import numpy.linalg as LA
import time
import pdb
#import SignifTesterMultipole as SIG
#sys.path.append('/panfs/roc/groups/6/kumarv/airans/expeditions/Saurabh/QuadPoleAnal/')
from multiprocessing import Pool
import itertools
   
# Generate a symmetric matrix with diagonal values = 1 and off-diagonal values to be <vals>
def make_sym_matrix(n,vals):
  m = np.zeros([n,n], dtype=np.double)
  xs,ys = np.triu_indices(n,k=1)
  m[xs,ys] = vals
  m[ys,xs] = vals
  m[np.diag_indices(n)] = 1
  return m
  

# Creates a directory path if it doesn't exist before
def mkdirnotex(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)

def get_lev_and_levg(NewSetCM):
    V,D = LA.eig(NewSetCM)
    LEV = np.min(V)
    LEV_minors = get_lev_minors(NewSetCM)
    MinLEV_minor =  np.min(LEV_minors)
    LEVG = MinLEV_minor-LEV
    return [LEV,LEVG]
  
# INPUT: InputLs, where
#<InputLs[0][0]> denotes min correlation in the desired correlation matrix  
#<InputLs[0][1]> denotes max correlation in the desired correlation matrix
#<InputLs[1][0]> denotes size of correlation matrix (no. of rows)  
#<InputLs[1][1]> denotes number of attempts made to generate a valid correlation matrix (positive semi-definite) for the given input parameters
# OUTPUT: 
# A list of 3x1 tuples, where each tuple = (least eigenvalue,linear gain,Min correlation) for each generated kxk random correlation matrix
def gen_dist_kxk_mats_edge_range(InputLs):
    min_edge = InputLs[0][0]
    max_edge = InputLs[0][1]
    sz = InputLs[1][0]
    num_attmp = InputLs[1][1]
    FinalOutput = []
    num_vals = sz*(sz-1)/2
    num_success = 0
    for i in range(num_attmp):
        if num_success>100:
            break
        
        vals = np.random.rand(num_vals)
        vals = vals/np.max(vals)
        span = max_edge-min_edge
        vals = min_edge + vals*span
        A = make_sym_matrix(sz,vals)
        V,D = LA.eig(A)        
        if np.min(V)>=0:       
            levind = np.argmin(V)
            EigVec = D[:,levind]
            NegInds = np.where(EigVec<0)
            A[:,NegInds] = A[:,NegInds]*(-1)
            A[NegInds,:] = A[NegInds,:]*(-1) 
            # Note: LEV stands for Least Eigenvalue = 1-linear dependence. LEVG = linear gain              
            [LEV,LEVG] = get_lev_and_levg(A) 
            if isinstance(LEV,complex) or isinstance(LEVG,complex):
                continue
            
            xs,ys = np.triu_indices(sz,k=1)
            MaxEdge = np.max(A[xs,ys])
            FinalOutput.append([LEV,LEVG,MaxEdge])
            num_success = num_success+1
    return FinalOutput
    


# INPUT:
# Parallelly generates correlation matrices of size <InputLs[1][0]> with <InputLs[0]> as minimum value of correlation
# <InputLs[1][1]> used in gen_dist_kxk_mats_edge_range()
# <InputLs[1][2]> indicates numerical resolution used in gen_dist_kxk_mats_edge_range()
#OUTPUT:
# A list of 3x1 tuples where each tuple = (least eigenvalue,linear gain,Min correlation) in a generated random correlation matrix
def gen_dist_kxk_mats_min_edge(InputLs):
    min_edge = InputLs[0]
    sz = InputLs[1][0]
    num_attmp = InputLs[1][1]
    num_resol = InputLs[1][2]
    All_nums = np.arange(min_edge,1,num_resol)
    print "min_edge = "+str(min_edge)+" sz=" + str(sz) + " num_attmp="+str(num_attmp)
    FinalOutput = []
    
    OtherInputs = [sz,num_attmp]
    for i in range(All_nums.size):
        max_edge = All_nums[i]
        EdgeRange = [min_edge,max_edge]
        InputLs = [EdgeRange,OtherInputs]
        Output = gen_dist_kxk_mats_edge_range(InputLs)
        FinalOutput = FinalOutput + Output
    return FinalOutput


# Main Code               
if __name__ == "__main__":
    sz = 6
#    num_samp = 10**5
    num_resol = 0.01
    num_proc = 20
    num_attmp = 500

    t1 = time.time() 
    OtherInputs = [sz,num_attmp,num_resol]  
    pool = Pool(processes=20)
        
    MinEdges = np.arange(-1,1,num_resol).tolist()
    FinalOutput = pool.map(gen_dist_kxk_mats_min_edge, itertools.izip(MinEdges,itertools.repeat(OtherInputs)))          
    pool.close()
    FinalOutput = sum(FinalOutput,[])
    t2 = time.time()
    print "Time elapsed:"+str(t2-t1)+"seconds"
    
    LEVs,LEVGs,MaxEdge = zip(*FinalOutput)
    LEVs = np.asarray(LEVs)
    LEVGs = np.asarray(LEVGs)
    MaxEdge = np.asarray(MaxEdge)
    saveData = {'LEVs':LEVs,'LEVGs':LEVGs,'MaxEdge':MaxEdge}
    
   
    savedir = os.getcwd()+'Empirical Results/';
    mkdirnotex(savedir)
    file_str = 'size_'+str(sz)+'_numattempts_'+str(num_attmp)   
    sio.savemat(savedir+'Empirical_Results_Sets_'+file_str,saveData,appendmat=True)

    
