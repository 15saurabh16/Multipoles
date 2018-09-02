# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 13:14:21 2018

@author: agraw066
"""

import Misc_Modules as MISC
import scipy.stats.mstats as statm
import os
import numpy as np
import scipy.io as sio
#import sys
import numpy.linalg as LA
import time
import pdb
import itertools
from multiprocessing import Pool
import COMET_ADVANCED as COMETA 
import math

def mkdirnotex(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)

def get_inputs_for_groups(InpList,GroupSz):
    NumMPs = len(InpList)
    GroupStInds = np.arange(0,NumMPs,GroupSz)
    NumGroups = GroupStInds.size
    GroupEndInds = GroupStInds + GroupSz
    GroupEndInds[GroupEndInds>NumMPs] = NumMPs
    AllGroups = [InpList[GroupStInds[i]:GroupEndInds[i]] for i in range(NumGroups)]
    return AllGroups
    
    
def brute_search(CorrMat,sz,levgth,sigma):
    AllLEVs = []
    AllLEVGs = []
    AllMPs = []
    num_ts = np.shape(CorrMat)[0]
    AllInds = range(num_ts)
    count = 0
    t1 = time.time()
    for i in itertools.combinations(AllInds,sz):
        count = count+1
        NewSet = list(i)
        NewCorrMat = CorrMat[np.ix_(NewSet,NewSet)]
        [lev,levg] = MISC.get_lev_and_levg(NewCorrMat)
        if lev<=1-sigma and levg>=levgth:
            AllMPs.append(NewSet)
            AllLEVs.append(lev)
            AllLEVGs.append(levg)
        if np.remainder(count,10000)==0:
            t2 = time.time()
            print "count = :"+str(count)+" Time Elapsed:" + str(t2-t1) + " seconds"
 
    return [AllMPs,AllLEVs,AllLEVGs]
    
def brute_search_group(InputLs):
    GrpCombs = InputLs[0]
    CorrMat = InputLs[1][0]
    levgth = InputLs[1][1]
    sigma = InputLs[1][2]
    AllLEVs = []
    AllLEVGs = []
    AllMPs = []
#    num_ts = np.shape(CorrMat)[0]
    t1 = time.time()
    for i in GrpCombs:
        NewSet = list(i)
        NewCorrMat = CorrMat[np.ix_(NewSet,NewSet)]
        [lev,levg] = MISC.get_lev_and_levg(NewCorrMat)
        if levg>=levgth and lev<=1-sigma:
            AllMPs.append(NewSet)
            AllLEVs.append(lev)
            AllLEVGs.append(levg)
        

#    print "Group Time Elapsed:" + str(t2-t1) + " seconds"
    OutputLs = zip(AllMPs,AllLEVs,AllLEVGs)
    return OutputLs
   
    
    
def random_search_parallel2(CorrMat,minsz1,minsz2,time_lim,AllSigma,AllDelta):
    AllLEVs = []
    AllLEVGs = []
    AllMPs = []
    sigma = min(AllSigma)
    levgth = min(AllDelta)
    num_ts = np.shape(CorrMat)[0]
    t1 = time.time() # NEW

    max_chunk_sz = 10**6
    chunk_sz = max_chunk_sz
    num_proc = 20
    pool = Pool(processes = num_proc)
    
    DeltaT = 0
    DeltaU = 0 # Last Update
    UpdateInt = 50
    LastUpdate = time.time()
    while (DeltaT<time_lim):
        ChnkCombs = []
        for i in range(chunk_sz):
            sz = np.random.choice(range(minsz1,minsz2+1))
            NewSet = list(np.random.choice(num_ts,sz,replace=True))       
            ChnkCombs.append(NewSet)
        GroupSz = 20000 #10**5
        print "GroupSz = " +str(GroupSz)
        AllGroups = []
        OtherInputs = [CorrMat,levgth,sigma]
        AllGroupOutput = pool.map(brute_search_group, itertools.izip(AllGroups,itertools.repeat(OtherInputs)))                      
        print "Chunk Finished"
        AllGroupCombined = sum(AllGroupOutput,[])
        if len(AllGroupCombined)>0:            
            ChunkMPsTup,ChunkLEVsTup,ChunkLEVGsTup = zip(*sum(AllGroupOutput,[]))
            AllMPs = AllMPs + list(ChunkMPsTup)
            AllLEVs = AllLEVs + list(ChunkLEVsTup)
            AllLEVGs = AllLEVGs + list(ChunkLEVGsTup)
        DeltaT = time.time() - t1
        DeltaU = time.time() - LastUpdate
        print "DeltaT = {}, NumMPs = {}".format(DeltaT,len(AllMPs))
        if DeltaU > UpdateInt:
            print "Update Time!!"
            generate_final_output(AllSigma,AllDelta,AllMPs,AllLEVs,AllLEVGs)
            LastUpdate = time.time()
        
    pool.close()       
    return [AllMPs,AllLEVs,AllLEVGs]


def generate_final_output(AllSigma,AllDelta,RndMPs,RndLEVs,RndLEVGs):
    NumRndMPs = []
    ParamCombos = []
    for s in range(len(AllSigma)):
        sigma = AllSigma[s]
        for d in range(len(AllDelta)):
            delta = AllDelta[d]
            FinalRndMPs = [RndMPs[i] for i in range(len(RndLEVs))  if (1-RndLEVs[i]>=sigma and RndLEVGs[i]>=delta)]
            FinalRndLEVs = [RndLEVs[i] for i in range(len(RndLEVs))  if (1-RndLEVs[i]>=sigma and RndLEVGs[i]>=delta)]
            FinalRndLEVGs = [RndLEVGs[i] for i in range(len(RndLEVs))  if (1-RndLEVs[i]>=sigma and RndLEVGs[i]>=delta)]
            [FinalRndMPs,FinalRndLEVs,FinalRndLEVGs,FinalRndSzList]= COMETA.remove_non_maximals(FinalRndMPs,FinalRndLEVs,FinalRndLEVGs,CorrMat,sigma,delta)            
            NumRndMPs.append(len(FinalRndMPs))
            ParamCombos.append([sigma,delta])        
            print "sigma = {}, delta = {}, no. of multipoles = {}".format(sigma,delta,len(FinalRndMPs))
            savedir = os.getcwd()+'/MultipolesClimate/'
            mkdirnotex(savedir)
            file_str1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
            file_str2 = '_sigma_'+str(sigma)+'_delta_'+str(delta)   
            saveData = {'FinalRndMPs':FinalRndMPs,'FinalRndLEVs':FinalRndLEVs,'FinalRndLEVGs':FinalRndLEVGs}            
            sio.savemat(savedir+'RANDSRCH_Multipoles_'+file_str1+file_str2,saveData,appendmat=True)
    return

    
if __name__ == '__main__':
    tau = 0.8
    loaddir = os.getcwd()
    data = sio.loadmat(loaddir + '/psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)+'.mat')
    InputTs = data['FinalTsData']
    InputTs = statm.zscore(InputTs,axis=0)
     
    AllSigma = [0.6,0.5,0.4]
    AllDelta = [0.2,0.15,0.1]
    minsz1 = 6
    minsz2 = 11
    time_lim = 22*3600
    t_beg = time.time()
    CorrMat = np.corrcoef(InputTs,rowvar=0)
    CorrMat = np.nan_to_num(CorrMat)
    t1 = time.time()
    [RndMPs,RndLEVs,RndLEVGs] = random_search_parallel2(CorrMat,minsz1,minsz2,time_lim,AllSigma,AllDelta)
    t2 = time.time()
    print "Random Search Completed for {} "+str(t2-t1)+" seconds"
    