# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:52:03 2017

@author: agraw066
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:54:32 2017

@author: agraw066
"""
import math
import numpy as np
import numpy.linalg as LA
from random import shuffle
import sys
#sys.path.append('/panfs/roc/groups/6/kumarv/airans/expeditions/Saurabh/QuadPoleAnal/')
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
import levg_null_dist_expt3 as NULL3
from multiprocessing import Pool

# removes redundant as well as non-maximal multipoles
def get_inputs_for_groups(InpList,GroupSz):
    NumMPs = len(InpList)
    GroupStInds = np.arange(0,NumMPs,GroupSz)
    NumGroups = GroupStInds.size
    GroupEndInds = GroupStInds + GroupSz
    GroupEndInds[GroupEndInds>NumMPs] = NumMPs
    AllGroups = [InpList[GroupStInds[i]:GroupEndInds[i]] for i in range(NumGroups)]
    return AllGroups
    
    
def remove_redundant_multipoles_alter(MPList,LEVList,LEVGList):
    if MPList==[]:
        return [[],[],[],[]]
    MPSizes = [len(MPList[i]) for i in range(len(MPList))]
    ZippedLs = zip(MPList,LEVList,LEVGList,MPSizes)
    ZippedLs.sort(key=lambda x: x[3],reverse = True) # sort in decreasing order of sizes
    MPTuple,LEVTuple,LEVGTuple,SzTuple = zip(*ZippedLs)
    NumMPs = len(MPTuple)
#    SimMat = np.empty((NumMPs,NumMPs),dtype=bool)
    # Note that SimMat is not a symmetric matrix. SimMat([i][j]==1 if ith multipole subsumes or is exactly same as jth multipole
    FinalMPList = []
    FinalLEVList = []
    FinalLEVGList = []
    FinalSzList = []
    IsRemoved = np.zeros((NumMPs,),dtype=bool)
    for i in range(NumMPs):
#        t1 = time.time()
        NewMP = MPTuple[i]
        if not IsRemoved[i]:
            FinalMPList.append(NewMP)
            FinalLEVList.append(LEVTuple[i])
            FinalLEVGList.append(LEVGTuple[i])
            FinalSzList.append(SzTuple[i])
            for j in range(i,len(MPTuple)):
                if not IsRemoved[j]:
                    IsRemoved[j] = set(MPTuple[j]).issubset(NewMP)
#        t2 = time.time()
#        print "Time Elapsed: " + str(t2-t1) + " seconds for i = " + str(i)
    return [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList]

def remove_redundant_multipoles_alter_group(InputLs):
    MPTuple,LEVTuple,LEVGTuple = zip(*InputLs)
    
    MPList = list(MPTuple)
    LEVList = list(LEVTuple)
    LEVGList = list(LEVGTuple)
    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = remove_redundant_multipoles_alter(MPList,LEVList,LEVGList)
    OutputLs = zip(FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList)
    return OutputLs

def remove_redundant_multipoles_alter_parallel(MPList,LEVList,LEVGList):
    num_proc = 20
    InpList = zip(MPList,LEVList,LEVGList)
    group_sz = min(100000,len(MPList)/num_proc)
    AllGroups = get_inputs_for_groups(InpList,group_sz)
    pool = Pool(processes = num_proc)
    AllGroupOutput = pool.map(remove_redundant_multipoles_alter_group, AllGroups)                      
    pool.close() 
    AllMPsTup,AllLEVsTup,AllLEVGsTup,AllSzsTup = zip(*sum(AllGroupOutput,[]))
    AllMPsList = list(AllMPsTup)
    AllLEVsList= list(AllLEVsTup)
    AllLEVGsList= list(AllLEVGsTup)
    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = remove_redundant_multipoles_alter(AllMPsList,AllLEVsList,AllLEVGsList)
    return [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList]

def remove_redundant_multipoles_alter_parallel_recursive(MPList,LEVList,LEVGList):
    JobDone = False
    num_proc = 20
    pool = Pool(processes = num_proc)
    while not JobDone: 
        NumMPs1 = len(MPList)
        InpList = zip(MPList,LEVList,LEVGList)
        group_sz = min(100000,len(MPList)/num_proc)
        print "NumMPs1 ={}, group_sz = {}".format(NumMPs1,group_sz)
        AllGroups = get_inputs_for_groups(InpList,group_sz)
        print "num_groups = {}".format(len(AllGroups))
        AllGroupOutput = pool.map(remove_redundant_multipoles_alter_group, AllGroups)                      
        AllMPsTup,AllLEVsTup,AllLEVGsTup,AllSzsTup = zip(*sum(AllGroupOutput,[]))
        MPList = list(AllMPsTup)
        LEVList= list(AllLEVsTup)
        LEVGList= list(AllLEVGsTup)
#        SzList = list(AllSzsTup)
        NumMPs2 = len(MPList)
        if NumMPs2 >= 0.9*NumMPs1:
            JobDone  = True
            
    pool.close()
    print "Final Round NumMPs = {}".format(len(LEVList))        
    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = remove_redundant_multipoles_alter(MPList,LEVList,LEVGList)    
    return  [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList]


def remove_redundant_multipoles(MPList,LEVList,LEVGList):
    ZippedLs = zip(MPList,LEVList,LEVGList)
    ZippedLs.sort(key=lambda x: x[1]) # sort in inreasing LEVs
    MPTuple,LEVTuple,LEVGTuple = zip(*ZippedLs)
    NumMPs = len(MPTuple)
#    SimMat = np.empty((NumMPs,NumMPs),dtype=bool)
    # Note that SimMat is not a symmetric matrix. SimMat([i][j]==1 if ith multipole subsumes or is exactly same as jth multipole
    t1 = time.time()
    SimMat= np.array([[len(set(MPTuple[j]) - set(MPTuple[i]))==0 for j in range(len(MPTuple))] for i in range(len(MPTuple))],dtype=bool)
    t2 = time.time()
    print "Time Elapsed: " + str(t2-t1) + " seconds"
    
    t1 = time.time()
    FinalZippedLs = []
    IsIncluded = np.zeros((NumMPs,),dtype=bool)
    
    # Include the multipole with lowest LEV and remove all the multipoles that are subsumed by it or its duplicates 
    for i in range(NumMPs):
        if IsIncluded[i]:
            continue
        FinalZippedLs.append((MPTuple[i],LEVTuple[i],LEVGTuple[i]))
        NbInds = np.where(SimMat[i][:])[0]
        IsIncluded[NbInds] = True
        SimMat[NbInds][:] = False
        SimMat[:][NbInds] = False
    t2 = time.time()
    print "Time Elapsed: " + str(t2-t1) + " seconds"    
    FinalMPList,FinalLEVList,FinalLEVGList = zip(*FinalZippedLs)
    FinalMPList = list(FinalMPList)
    FinalMPList = [list(x) for x in FinalMPList]
    FinalLEVList = list(FinalLEVList)
    FinalLEVGList = list(FinalLEVGList)
    return [FinalMPList,FinalLEVList,FinalLEVGList]


def get_lev_minors(InpCM):
    NumVars = InpCM.shape[0]
    AllLEVMinors = np.zeros((NumVars,))
    MemList = range(NumVars)
    for i in MemList: 
        SelMems = MemList[:]       
        SelMems.pop(i)            
        PMinor = InpCM[np.ix_(SelMems,SelMems)]
        V,D = LA.eig(PMinor)
        AllLEVMinors[i] = np.min(V)
    return AllLEVMinors


def get_lev_and_levg(NewSetCM):
    V,D = LA.eig(NewSetCM)
    LEV = np.min(V)
    LEV_minors = get_lev_minors(NewSetCM)
    MinLEV_minor =  np.min(LEV_minors)
    LEVG = MinLEV_minor-LEV
    return [LEV,LEVG]

def get_lev_levg_and_weakest_pole(NewSetCM):
    V,D = LA.eig(NewSetCM)
    LEV = np.min(V)
    LEV_minors = get_lev_minors(NewSetCM)
    MinLEV_minor =  np.min(LEV_minors)
    WeakestPole = np.argmin(LEV_minors)
    LEVG = MinLEV_minor-LEV
    return [LEV,LEVG,WeakestPole]
    
def key_generator(InpList,mult_fac):
    key = 0
    count =0
    InpList.reverse()
    for i in InpList:
#        print str(i)
        key+=math.pow(mult_fac,count)*i
        count+=1
    return int(key)
    
def find_multipoles_for_P2(InputLs):
    P2 = InputLs[0]
    OtherInputs = InputLs[1]
    CandP1s = OtherInputs[0]
    CorrMat = OtherInputs[1]
    delta = OtherInputs[2]
    NewP2MPs = []
    for j in range(len(CandP1s)):
        P1 = CandP1s[j]
        if not(list(set(P1) & set(P2))): # Ensuring that two sets are disjoint
            NewSet = list(set(P1+P2))
            NewSet.sort()
            if True: 
                NewSetCM = CorrMat[np.ix_(NewSet,NewSet)]# Extract correlation matrix
                [LEV,LEVG] = get_lev_and_levg(NewSetCM)
              
                if LEVG>=delta: # can also use alternate threshold delta2
                    NewP2MPs.append([tuple(NewSet),LEV,LEVG])
                    
    return NewP2MPs


def find_next_level_multipoles(CorrMat,inp_level,sigma,delta,AllMPs):
    NumPs = np.shape(CorrMat)[0]
    NewLEVs = []
    NewLEVGs = []
#    NewGoodMPs = []
    inp_sz = inp_level+1
    if inp_sz==1:          
        NewMPs = [(i,) for i in range(NumPs)] #range(NumPs)
#        shuffle(NewMPs)
        print "inp_level={}".format(inp_level)
        NewLEVs = [1]*NumPs
        NewLEVGs = [float('Inf')]*NumPs
        return [NewLEVs,NewLEVGs,NewMPs]
    
    
    NewMPs = []
    AllOutputP2 = []
    pool = Pool(processes=20)
    for i in range(inp_sz/2):
        ind1 = i
        sz1 = i+1
        sz2 = inp_sz-sz1
        ind2 = sz2 -1
        CandP1s = AllMPs[ind1]
        OtherInputs = [CandP1s,CorrMat,delta]      
        OutputP2 = pool.map(find_multipoles_for_P2, itertools.izip(AllMPs[ind2], itertools.repeat(OtherInputs)))          
        # Write a function to combine  multipoles from all P2s
        OutputP2 = sum(OutputP2,[])
        AllOutputP2.append(OutputP2)
#        OutputP2.sort(key=lambda x: x[2])
        
    pool.close() 
    AllOutputP2 = sum(AllOutputP2,[])
    AllOutputP2 = list(set(map(tuple,AllOutputP2)))
#    AllOutputP2 = [ list(x) for x in AllOutputP2]
    if len(AllOutputP2)>0:        
        NewMPs,NewLEVs,NewLEVGs = zip(*AllOutputP2)
#    pdb.set_trace()
    # Write a function to combine
    return [NewLEVs,NewLEVGs,NewMPs]
    


# Bottom-Up Discovery Algorithm (Based on Partition Lemma)
#def BUD(CorrMat,max_size,sigma,delta):
#    
#    AllMPs = [[] for i in range(max_size)]
##    AllGoodMPs = [[] for i in range(max_size)]
#    AllLEVs =  [[] for i in range(max_size)]
#    AllLEVGs =  [[] for i in range(max_size)]
#    for level in range(max_size):
#        t1 = time.time()
#        [NewLEVs,NewLEVGs,NewMPs] = find_next_level_multipoles(CorrMat,level,sigma,delta,AllMPs)
#        t2 = time.time()
#        if t2-t1>0.01:
#            print "Time Elapsed for level {} ={}".format(level,t2-t1)
#            
##        AllGoodMPs[level] = NewGoodMPs
#        AllLEVs[level] = NewLEVs
#        AllLEVGs[level] = NewLEVGs
#        AllMPs[level] = NewMPs
#    
#    return [AllLEVs,AllLEVGs,AllMPs]
  
def object_to_list_arr(ObjArr): # used to convert the FinalMPList object array to listoflists
    ListArr = []
    ListArr= [list(ObjArr[0][i][0]) for i in range(ObjArr.size)]
    return ListArr
    
#if __name__ == '__main__':
#    tau = 0.8
#    loaddir = '/panfs/roc/groups/6/kumarv/agraw066/airans/expeditions/Saurabh/QuadPoleAnal/MatlabScripts/'
#    data = sio.loadmat(loaddir + 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)+'.mat')
#    InputTs = data['FinalTsData']
##    InputTs = np.random.randn(400,1000)
#    InputTs = statm.zscore(InputTs,axis=0)
#    CorrMat = np.corrcoef(InputTs,rowvar=0)
#    CorrMat = np.nan_to_num(CorrMat)
#    sigma = 0.1
#    delta = 0.15
##    TempMat = CorrMat[0:10][0:10]
#    [AllLEVs,AllLEVGs,AllMPs] = BUD(CorrMat,np.shape(CorrMat)[0],sigma,delta)
#    
#    AllMPs = [list(x) for x in AllMPs ]
#    AllLEVs = [list(x) for x in AllLEVs]
#    AllLEVGs = [list(x) for x in AllLEVGs]
#    MPList = sum(AllMPs[2:],[])
#    LEVList = sum(AllLEVs[2:],[])
#    LEVGList = sum(AllLEVGs[2:],[])
#    
#    [FinalMPList,FinalLEVList,FinalLEVGList] = remove_redundant_multipoles(MPList,LEVList,LEVGList)
#    
#    from os.path import expanduser
#    home = expanduser("~")
#    savedir = home+'/airans/expeditions/Saurabh/QuadPoleAnal/MultipolesClimate/';
#    file_str1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
#    file_str2 = 'sigma_'+str(sigma)+'_delta_'+str(delta)
#
#    saveData = {'FinalMPList':FinalMPList,'FinalLEVList':FinalLEVList,'FinalLEVGList':FinalLEVGList}
#    sio.savemat(savedir+'Multipoles_'+file_str1+file_str2,saveData,appendmat=True)
#    
#    loadData = sio.loadmat(savedir+'Multipoles_'+file_str1+file_str2)
#    FinalMPList = loadData['FinalMPList']
#    FinalMPList = object_to_list_arr(FinalMPList)
#    FinalLEVList = list(loadData['FinalLEVList'][0])
#    FinalLEVGList = list(loadData['FinalLEVGList'][0])
#
#    num_rand = 1000
#    [AllStYrs,AllEndYrs,AllWinPvals,AllWinLEVs,AllWinLEVGs] = SIG.pval_anal_SLP(FinalMPList,tau,num_rand)
#   
##    loadData = sio.loadmat(savedir+'Multipoles_'+file_str1+file_str2)
##    AllWinPvals = loadData['AllWinPvals']
#    saveData = {'FinalMPList':FinalMPList,'FinalLEVList':FinalLEVList,'FinalLEVGList':FinalLEVGList,'AllStYrs':AllStYrs,
#                'AllEndYrs':AllEndYrs,'AllWinPvals':AllWinPvals,'AllWinLEVs':AllWinLEVs,'AllWinLEVGs':AllWinLEVGs}
#    sio.savemat(savedir+'Multipoles_'+file_str1+file_str2,saveData,appendmat=True)
#    
#    num_rand3 = 10**5
#    InputSzs = [3,4,5,6]
#    [Null3LEVs,Null3LEVGs] = NULL3.pval_anal3_SLP(tau,InputSzs,num_rand3)    
#    saveData3 = {'InputSzs':InputSzs,'Null3LEVs':Null3LEVs,'Null3LEVGs':Null3LEVGs}
#    sio.savemat(savedir+'NullDistExpt3_'+file_str1+file_str2,saveData3,appendmat=True)