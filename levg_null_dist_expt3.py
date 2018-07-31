# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:42:23 2017

@author: agraw066
"""

import math
import numpy as np
import numpy.linalg as LA
from random import shuffle
import sys
sys.path.append('/panfs/roc/groups/6/kumarv/airans/expeditions/Saurabh/QuadPoleAnal/')
#import pickle as pk
import scipy.io as sio
#import scipy.sparse.csgraph as sig
import time
import os
#import get_reg
import pdb  # Debugger
import scipy.stats.mstats as statm
#import BST
import itertools
import SignifTesterMultipole as SIG
from multiprocessing import Pool
import random 
import Misc_Modules as MISC

def get_null_dist_expt3(InputSzList,AllDataSets,num_rand):
    AllRandLEVs = np.zeros((num_rand,len(InputSzList)))
    AllRandLEVGs = np.zeros((num_rand,len(InputSzList)))
    NumDataSets = len(AllDataSets)
    TsLen = AllDataSets[0].shape[0]
    NumTsPerSet= AllDataSets[0].shape[1]
    for i in range(len(InputSzList)):
        sz = InputSzList[i]
        for j in range(num_rand):
            SelDSInds = random.sample(range(NumDataSets),sz)
            RandMPTs = np.zeros([TsLen,sz])
            for k in range(sz):
                TsData = AllDataSets[SelDSInds[k]]
                RandMPTs[:,k] = TsData[:,random.randint(0,NumTsPerSet-1)]
                
            RandCMat = np.corrcoef(RandMPTs,rowvar=False)
            [AllRandLEVs[j][i],AllRandLEVGs[j][i]] = MISC.get_lev_and_levg(RandCMat)
    
    return [AllRandLEVs,AllRandLEVGs]
    

def pval_anal3_SLP(tau,InputSzList,num_rand):
    loaddir = os.getcwd()
    loadfile = loaddir+'/AllWins_psl_C12_73x144_0.8_50_'+str(tau)+'_1900_36_1975.mat'
    DataInfo = sio.loadmat(loadfile)
    AllWinFinalTsData = DataInfo['AllWinFinalTsData']
    
    AllDatasets = []
    for i in range(AllWinFinalTsData.shape[0]):
        AllDatasets.append(AllWinFinalTsData[i][0])
    t1 = time.time()
    [NullLEVs,NullLEVGs] = get_null_dist_expt3(InputSzList,AllDatasets,num_rand)
    t2 = time.time()
    print "Time taken for Expt3:{} seconds".format(t2-t1)
    return [NullLEVs,NullLEVGs]

def pval_anal3_fMRI(State,InputSzList,num_rand):
    loaddir = os.getcwd()
    data  = sio.loadmat(loaddir+'/AllScanData'+State+'State.mat')
    
    if State == 'Rest':
        InputTs = np.transpose(data['AllScansRest'][:])
    else:
        InputTs = np.transpose(data['AllScansCartoon'][:])
    
    AllDatasets = []
    for i in range(InputTs[0].size):
        AllDatasets.append(np.transpose(InputTs[0][i]))
        
    t1 = time.time()    
    [NullLEVs,NullLEVGs] = get_null_dist_expt3(InputSzList,AllDatasets,num_rand)
    t2 = time.time()
    print "Time taken for Expt3:{} seconds".format(t2-t1)
    return [NullLEVs,NullLEVGs]

