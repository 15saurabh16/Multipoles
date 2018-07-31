# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 21:59:52 2017

@author: agraw066
"""


import math
import numpy as np
import numpy.linalg as LA
from random import shuffle,sample
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
from multiprocessing import Pool
import Misc_Modules as MISC

def get_multiple_groups(FinalMPList,GroupSz):
    NumMPs = len(FinalMPList)
    GroupStInds = np.arange(0,NumMPs,GroupSz)
    NumGroups = GroupStInds.size
    GroupEndInds = GroupStInds + GroupSz
    GroupEndInds[GroupEndInds>NumMPs] = NumMPs
    AllGroups = [FinalMPList[GroupStInds[i]:GroupEndInds[i]] for i in range(NumGroups)]
    return AllGroups

def get_pval_multipole_group(InputLs):
    MPList = InputLs[0]
    OtherInputs = InputLs[1]
    TsData = OtherInputs[0]
    RandTsData= OtherInputs[1]
    num_rand = OtherInputs[2]
    NumMPs = len(MPList)
    pval_list = np.zeros((NumMPs,))
    for i in range(NumMPs):
        Multipole = MPList[i]
        pval_list[i] = get_pval_multipole(Multipole,TsData,RandTsData,num_rand)
    return pval_list


# RandTsData should not include any time series from TsData
def get_pval_multipole(Multipole,TsData,RandTsData,num_rand):
    NumObs = TsData.shape[0]
    MPCMat = np.corrcoef(TsData[np.ix_(np.arange(NumObs),Multipole)],rowvar=False)
    [LEV_orig,LEVG_orig,WeakestPole] = MISC.get_lev_levg_and_weakest_pole(MPCMat)
    MemList = range(len(Multipole))
    SelMems = MemList[:]       
    SelMems.pop(WeakestPole)
    
    MPTs1 = np.zeros((NumObs,len(Multipole)))
    StrongPoleInds = [Multipole[x] for x in SelMems]
    MPTs1[np.ix_(np.arange(NumObs),SelMems)] = TsData[np.ix_(np.arange(NumObs),StrongPoleInds)]
    
    NumRandTs = RandTsData.shape[1]
    SelInds = sample(range(NumRandTs),num_rand)
#    pdb.set_trace()
    SelRandTs = RandTsData[np.ix_(range(NumObs),SelInds)]
    
#    RandMPCMat = np.eye(len(Multipole))
#    RandMPCMat[np.ix_(SelMems,SelMems)] = MPCMat[np.ix_(SelMems,SelMems)]
    AllRandLEVG = np.zeros((num_rand,))
    for i in range(num_rand):
        MPTs1[:,WeakestPole] = SelRandTs[:,i]
        RandMPCMat = np.corrcoef(MPTs1,rowvar=False)
        [_,AllRandLEVG[i]] = MISC.get_lev_and_levg(RandMPCMat)
    
    pval = np.where(AllRandLEVG>=LEVG_orig)[0].size/float(num_rand)
#    pdb.set_trace()
    return pval
    
    
def get_lev_and_levg_group(InputLs):
    MPList = InputLs[0]
    OtherInputs = InputLs[1]
    TsData = OtherInputs[0]
    NumObs = TsData.shape[0]
    NumMPs = len(MPList)
    LEVs = np.zeros((NumMPs,))
    LEVGs = np.zeros((NumMPs,))
    
    for i in range(NumMPs):
        MPTs = TsData[np.ix_(range(NumObs),MPList[i])]
        NewSetCM = np.corrcoef(MPTs,rowvar=False) 
        [LEVs[i],LEVGs[i]] = MISC.get_lev_and_levg(NewSetCM)

    GroupLEVandLEVGs = zip(LEVs,LEVGs) #[(lev1,levg1),(lev2,levg2),(lev3,levg3)]....
    return GroupLEVandLEVGs


def get_lev_and_levg_all_wins(FinalMPList,AllDatasets):
    NumMPs = len(FinalMPList)
    NumDatasets = len(AllDatasets)    
    num_proc = 20
    group_sz = max(len(FinalMPList)/num_proc,1)
    AllGroups = get_multiple_groups(FinalMPList,group_sz)
    AllWinLEVs = np.zeros((NumMPs,NumDatasets))
    AllWinLEVGs = np.zeros((NumMPs,NumDatasets))

    for i in range(NumDatasets):
        TsData = AllDatasets[i]
        OtherInputs = [TsData]
#        t1 = time.time()
        pool = Pool(processes = num_proc)
        AllGroupLEVandLEVGs = pool.map(get_lev_and_levg_group, itertools.izip(AllGroups, itertools.repeat(OtherInputs)))                      
        pool.close() 
#        t2 = time.time()
#        print "Time Elapsed: {} seconds".format(t2-t1) 
        AllLEVsTup,AllLEVGsTup = zip(*sum(AllGroupLEVandLEVGs,[]))
        AllWinLEVs[:,i]  = np.array(list(AllLEVsTup))
        AllWinLEVGs[:,i] = np.array(list(AllLEVGsTup))
     
        #Expand and then unzip ALlGroupLEVandLEVGs
    return [AllWinLEVs,AllWinLEVGs]
    
    
def pval_parallel(FinalMPList,AllDatasets,num_rand):
    NumMPs = len(FinalMPList)
    NumDatasets = len(AllDatasets)
    NumTsPerDataset = AllDatasets[0].shape[1]
    AllWinPvals = np.zeros((NumMPs,NumDatasets))

    # Step 1: Create Groups of Multipoles for parallelizing purposes
    num_proc = 20
    group_sz = max(len(FinalMPList)/num_proc,1)
    AllGroups = get_multiple_groups(FinalMPList,group_sz)
    
    # Step 2: Create Giant Matrix storing Ts of all datasets
    AllDatasetTs = AllDatasets[0]
    for i in range(1,NumDatasets):
        AllDatasetTs = np.concatenate((AllDatasetTs,AllDatasets[i]),axis=1)
    
    for i in range(NumDatasets):
        RandTsData = np.delete(AllDatasetTs,np.s_[NumTsPerDataset*i:NumTsPerDataset*(i+1)],axis=1)
        OtherInputs = [AllDatasets[i],RandTsData,num_rand]
        t1 = time.time()
        pool = Pool(processes = num_proc)
        AllGroupPvals = pool.map(get_pval_multipole_group, itertools.izip(AllGroups, itertools.repeat(OtherInputs)))                      
        pool.close()
        t2 = time.time()
        print "Time Elapsed: {} seconds".format(t2-t1)
        #Step 4: Concatenate pvalues from all groups to a single vector and update AllWinPvals 
        AllWinPvals[:,i] = np.concatenate(tuple(AllGroupPvals),axis=0)

    return [AllWinPvals]
    
def pval_anal_model_SLP(InputMPs,tau,model_name,num_rand):
    loaddir = '/panfs/roc/groups/6/kumarv/agraw066/airans/expeditions/Saurabh/QuadPoleAnal/MultipolesClimate/'
    loadfile = loaddir+'AllWins_'+model_name+'_psl_C12_73x144_0.8_50_'+str(tau)+'_1900_36_1975.mat'
    DataInfo = sio.loadmat(loadfile)
    AllModelStYrs = DataInfo['AllStYrs']
    AllModelEndYrs = DataInfo['AllEndYrs']
    AllWinFinalTsData = DataInfo['AllWinFinalTsData']
    
    AllDatasets = []
    for i in range(AllWinFinalTsData.shape[0]):
        AllDatasets.append(AllWinFinalTsData[i][0])
    
    [ModelLEVs,ModelLEVGs] = get_lev_and_levg_all_wins(InputMPs,AllDatasets)
    [ModelPvals] = pval_parallel(InputMPs,AllDatasets,num_rand) # TEMPORARY EXCLUDED
    
    from os.path import expanduser
    home = expanduser("~")
    savedir = home+'/airans/expeditions/Saurabh/QuadPoleAnal/MultipolesClimate/'
    savefilename = 'Model_Analysis_'+model_name+'_KDD_multipole.mat'
    saveData = {'AllModelStYrs':AllModelStYrs,'AllModelEndYrs':AllModelEndYrs,'ModelPvals':ModelPvals,'ModelLEVs':ModelLEVs,'ModelLEVGs':ModelLEVGs}
    sio.savemat(savedir+savefilename,saveData,appendmat=True)
    return [AllModelStYrs,AllModelEndYrs,ModelPvals,ModelLEVs,ModelLEVGs]

    
def pval_anal_SLP(FinalMPList,tau,num_rand):
    loaddir = os.getcwd()
    loadfile = loaddir+'/AllWins_psl_C12_73x144_0.8_50_'+str(tau)+'_1900_36_1975.mat'
    DataInfo = sio.loadmat(loadfile)
    AllStYrs = DataInfo['AllStYrs']
    AllEndYrs = DataInfo['AllEndYrs']
    AllWinFinalTsData = DataInfo['AllWinFinalTsData']
    
    AllDatasets = []
    for i in range(AllWinFinalTsData.shape[0]):
        AllDatasets.append(AllWinFinalTsData[i][0])
    
    [AllWinLEVs,AllWinLEVGs] = get_lev_and_levg_all_wins(FinalMPList,AllDatasets)
    [AllWinPvals] = pval_parallel(FinalMPList,AllDatasets,num_rand) # TEMPORARY EXCLUDED
#    AllWinPvals = [] # TO BE REMOVED
    return [AllStYrs,AllEndYrs,AllWinPvals,AllWinLEVs,AllWinLEVGs]

    
def pval_anal_fMRI(FinalMPList,State,num_rand):
    loaddir = os.getcwd()
    data  = sio.loadmat(loaddir+'/AllScanData'+State+'State.mat')
    
    if State == 'Rest':
        InputTs = np.transpose(data['AllScansRest'][:])
    else:
        InputTs = np.transpose(data['AllScansCartoon'][:])
    
    AllDatasets = []
    for i in range(InputTs[0].size):
        AllDatasets.append(np.transpose(InputTs[0][i]))
        
    [AllWinLEVs,AllWinLEVGs] = get_lev_and_levg_all_wins(FinalMPList,AllDatasets)
    [AllWinPvals] = pval_parallel(FinalMPList,AllDatasets,num_rand) # TEMPORARY EXCLUDED
#    AllWinPvals = [] # TO BE REMOVED
    return [AllWinPvals,AllWinLEVs,AllWinLEVGs]

    

    