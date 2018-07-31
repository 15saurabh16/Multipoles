# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:00:15 2018

@author: agraw066
"""


import Misc_Modules as MISC
import scipy.stats.mstats as statm
import os
import numpy as np
import scipy.io as sio
import sys
import numpy.linalg as LA
import time
import pdb
import bronk_kerb as bk
import SignifTesterMultipole as SIG
import levg_null_dist_expt3 as NULL3
import itertools
import CLIQUE_multipoles_algorithm as CLIQ
#import BruteForceSearch as BRUT
from multiprocessing import Pool


def get_inputs_for_groups(InpList,GroupSz):
    NumMPs = len(InpList)
    GroupStInds = np.arange(0,NumMPs,GroupSz)
    NumGroups = GroupStInds.size
    GroupEndInds = GroupStInds + GroupSz
    GroupEndInds[GroupEndInds>NumMPs] = NumMPs
    AllGroups = [InpList[GroupStInds[i]:GroupEndInds[i]] for i in range(NumGroups)]
    return AllGroups


def cliq_brute_search(Cliq,CliqMat,cliq_sz,sigma,levgth):
    AllLEVs = []
    AllLEVGs = []
    AllMPs = []
    num_ts = np.shape(CliqMat)[0]
    AllInds = range(num_ts)
    count = 0
    CliqArr = np.array(Cliq)
    for subset in itertools.combinations(AllInds,cliq_sz):
        NewSet = CliqArr[np.ix_(subset)]        
        count = count+1
        NewCorrMat = CliqMat[np.ix_(subset,subset)]               
        sz = len(NewSet)
        W = np.abs(NewCorrMat) - np.eye(sz)           
        mean_vec = np.divide(np.mean(W,axis=1)*sz,(sz-1)*1.0)
        min_wt = np.min(mean_vec)
        if min_wt>=levgth:
            [lev,levg] = MISC.get_lev_and_levg(NewCorrMat)
            if levg>=levgth and lev<=1-sigma:
                AllMPs.append(NewSet.tolist())
                AllLEVs.append(lev)
                AllLEVGs.append(levg)
 
    return [AllMPs,AllLEVs,AllLEVGs]
    
def extract_multipoles_from_cliq(Cliq,CliqMat,sigma,delta):
    
    CliqMPs = []
    CliqLEVs = []
    CliqLEVGs = []  
    CliqSzs = []
    num_ts = np.shape(CliqMat)[0]
    max_sz = int(np.min([num_ts,np.floor(np.divide(1,delta)+1)]))
    for i in range(3,max_sz+1):
        [NewMPs,NewLEVs,NewLEVGs] = cliq_brute_search(Cliq,CliqMat,i,sigma,delta)
        CliqMPs = CliqMPs + NewMPs
        CliqLEVs = CliqLEVs + NewLEVs
        CliqLEVGs = CliqLEVGs + NewLEVGs
        CliqSzs = CliqSzs + [i]*len(NewMPs)
    return [CliqMPs,CliqLEVs,CliqLEVGs,CliqSzs]

          
def extract_multipoles_from_cliqgroup(InputLs):
    
    t1 = time.time()
    GrpInputs = InputLs[0] #[[cliq1,cliqmat1],[cliq2,cliqmat2],...]
    delta = InputLs[1][0]
    sigma = InputLs[1][1]
    
    GrpOutputMPs = []
    GrpOutputLEVs = []
    GrpOutputLEVGs = []
    GrpOutputSizes = []
    
    for i in range(len(GrpInputs)):
        cliq = GrpInputs[i][0]
        NewCorrMat = GrpInputs[i][1]
        [lev,levg] = MISC.get_lev_and_levg(NewCorrMat)
        if lev<=1-sigma and levg>=delta:
            NewMPs = [cliq]
            NewLEVs = [lev]
            NewLEVGs = [levg]
            NewSizes = [len(cliq)]
        elif lev<=1-sigma:
            [NewMPs,NewLEVs,NewLEVGs,NewSizes] = extract_multipoles_from_cliq(cliq,NewCorrMat,sigma,delta)
        else:
            continue
        
        GrpOutputMPs = GrpOutputMPs + NewMPs
        GrpOutputLEVs = GrpOutputLEVs + NewLEVs
        GrpOutputLEVGs = GrpOutputLEVGs + NewLEVGs
        GrpOutputSizes = GrpOutputSizes + NewSizes
    
    
    FinalGrpOutput = zip(GrpOutputMPs,GrpOutputLEVs,GrpOutputLEVGs,GrpOutputSizes)
    t2 = time.time()
    print "Time taken for the group: "+str(t2-t1) + " seconds"
    return FinalGrpOutput

   
def find_good_multipoles_complete_parallel(cliques_gf,CorrMat,sigma,delta,group_sz):
    
    AllLEVs = [] #np.zeros([len(cliques_gf),])
    AllLEVGs = [] #np.zeros([len(cliques_gf),])
    AllSizes = [] #np.zeros([len(cliques_gf),])
    AllMPs = [] #[[] for i in range(len(cliques_gf))]    
    num_proc = 20
    InpList = []
    
    for i in range(len(cliques_gf)):
        cliq = list(cliques_gf[i])
        NewCorrMat = CorrMat[np.ix_(cliq,cliq)]
        InpList.append([cliq,NewCorrMat])
    print("Input List Created")
    AllGroups = get_inputs_for_groups(InpList,group_sz)
    pool = Pool(processes = num_proc)
    OtherInputs = [delta,sigma]
    AllGroupOutput = pool.map(extract_multipoles_from_cliqgroup, itertools.izip(AllGroups, itertools.repeat(OtherInputs)))                      
    pool.close() 
    AllMPsTup,AllLEVsTup,AllLEVGsTup,AllSzsTup = zip(*sum(AllGroupOutput,[]))
    AllMPs = list(AllMPsTup)
    AllLEVs  = list(AllLEVsTup)
    AllLEVGs = list(AllLEVGsTup)
    AllSizes = list(AllSzsTup)
    return [AllMPs,AllLEVs,AllLEVGs,AllSizes]  


def COMET_EXT(CorrMat,sigma,delta,edge_filt,group_sz):
    t_st = time.time()
    num_ts = np.shape(CorrMat)[0]
    # Construction of Negative Graph
    CorrMat1 = np.copy(CorrMat)
    CorrMat1[CorrMat1>=edge_filt] = 3
    CorrMat1[CorrMat1<=edge_filt] = 1
    CorrMat1[CorrMat1==3] = 0
    
   # Graph1 of Negative Edges
    Graph1 = {}
    for i in range(num_ts):
        Graph1[i] = np.where(CorrMat1[i][:]==1)[0].tolist()
  

    #CorrMat2: Only retains positive edges
    CorrMat2 = np.copy(CorrMat)
    CorrMat2[CorrMat<=-edge_filt] = 0
    CorrMat2[CorrMat>=-edge_filt] = 1
    CorrMat2[CorrMat>0.99] = 0

    #Graph2: for CorrMat2 
    Graph2 = {} # Only retains positive edges
    for i in range(num_ts):
        Graph2[i] = np.where(CorrMat2[i][:]==1)[0].tolist()
   
    #GraphF: Union (Graph1,Graph2) alongwith all cross-connections being the positive edges
    GraphF = CLIQ.construct_GraphF(Graph1,Graph2,num_ts)

    # Find negative/pseudo-negative cliques in GraphF
    t1 = time.time()
    cliques_tmp = bk.find_cliques(GraphF)
    t2 = time.time()
    print "Time Elapsed in generating cliques of GraphF:"+str(t2-t1) + " seconds"
    print "Total Cliques Obtained: "+str(len(cliques_tmp))   
    
    # Remove Duplicate cliques 
#    cliques_tmp2 = CLIQ.remove_g2_cliques(cliques_tmp,num_ts)
    cliques_gf = CLIQ.remove_duplicate_cliqs(cliques_tmp,num_ts)   
    print "Total Cliques Obtained After Removing Duplicate Cliques: "+str(len(cliques_gf))   
#    pdb.set_trace()
    # Find multipoles with strong linear gain from the obtained cliques
    t1 = time.time()
#    group_sz = 50000
    [AllMPs,AllLEVs,AllLEVGs,AllSizes] = find_good_multipoles_complete_parallel(cliques_gf,CorrMat,sigma,delta,group_sz)
    t2 = time.time()
    print "Time Elapsed in finding good MPs:"+str(t2-t1) + " seconds"

#    pdb.set_trace()
    # REMOVE DUPLICATE MULTIPOLES
    t1 = time.time()
    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = MISC.remove_redundant_multipoles_alter_parallel(AllMPs,AllLEVs,AllLEVGs)
#    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = MISC.remove_redundant_multipoles_alter(AllMPs,AllLEVs,AllLEVGs)
    t2 = time.time()
    print "Time Elapsed in eliminating non-maximal/duplicate MPs:"+str(t2-t1) + " seconds"
    
    t_end = time.time()
    print "Total Time for CLIQ COMPLETE: " + str(t_end - t_st) + "seconds"
    return [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList]
    

# TRIAL CODE   
#CurrMat = NewCorrMat
#t1 = time.time()
#[lev,levg] = MISC.get_lev_and_levg(CurrMat)
#t2 = time.time()
#print "Time for levg computation: "+str(t2 -t1)
#
#sz = np.shape(CurrMat)[0]
#t1 = time.time()
#W = np.abs(CurrMat) - np.eye(sz)           
#mean_vec = np.divide(np.mean(W,axis=1)*sz,(sz-1)*1.0)
#min_wt = np.min(mean_vec)
##min_ind = np.argmin(mean_vec)
#t2 = time.time()
#print "Time for mean computation: "+str(t2 -t1)
