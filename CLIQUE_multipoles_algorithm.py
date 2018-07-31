# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:53:10 2017

@author: agraw066
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:22:00 2017

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




#def break_clique(InpMat):
#    InpMat = P1
#    sz = np.shape(InpMat)[0]
#    W = np.abs(InpMat) - np.eye(sz)    
#    deg_vec = np.sum(W,axis=1)
#    D = np.diag(deg_vec)
#    L = D-W
#    EIGVALS,EIGVECS = LA.eig(L)

def get_supersets(InpLs,cliques):
    SuperSets = []
    SuperSetInds = []
    for i in range(len(cliques)):
        if set(InpLs).issubset(cliques[i]):
            SuperSets.append(cliques[i])
            SuperSetInds.append(i)
    return SuperSets,SuperSetInds


def construct_GraphF(Graph1,Graph2,num_ts):
    GraphF = {}
    for i in range(num_ts):
        Nbs1 = np.array(Graph1[i])
        Nbs2 = np.array(Graph2[i]) + num_ts #np.array([x+num_ts for x in Graph2[i]])
#        Nbs2 = Nbs2[Nbs2>i+num_ts]
        GraphF[i] = Nbs1.tolist() + Nbs2.tolist()
    
    for i in range(num_ts):
        Nbs1 = np.array(Graph1[i]) + num_ts
        Nbs2 = np.array(Graph2[i])
#        Nbs2 = Nbs2[Nbs2<i]
        GraphF[i+num_ts] = Nbs2.tolist() +  Nbs1.tolist()
    return GraphF


def remove_neg_cliques(cliques_tmp,num_ts):
    cliques_gf = []
    count_pseudo = 0
    for i in range(len(cliques_tmp)):
        cliq = np.array(list(cliques_tmp[i]))      
        if np.min(cliq)<num_ts and np.max(cliq)>=num_ts:
            count_pseudo+= 1
            cliq[cliq>=num_ts] = cliq[cliq>=num_ts]-num_ts
            cliques_gf.append(cliq.tolist())
    return cliques_gf

    
def remove_g2_cliques(cliques_tmp,num_ts):
    cliques_gf = []
    count_cliq = 0
    for i in range(len(cliques_tmp)):
        cliq = np.array(list(cliques_tmp[i]))      
        if np.min(cliq)<num_ts :
            count_cliq+= 1
            cliq[cliq>=num_ts] = cliq[cliq>=num_ts]-num_ts
            cliques_gf.append(cliq.tolist())
    return cliques_gf


# Used in scalability code: cliques_tmp is a list of lines of output of quick_cliques 
def remove_duplicate_cliqs_2(cliques_tmp,num_ts):
    cliques_gf = []
    count_cliq = 0
    for i in range(len(cliques_tmp)):
        CliqNodes = cliques_tmp[i].split(' ')
        for j in range(len(CliqNodes)-1):
            CliqNodes[j] = int(CliqNodes[j])
        CliqNodes[-1] = int(CliqNodes[-1].split('\n')[0])
        cliq = np.array(list(CliqNodes))
        c1 = np.min(cliq)
        
        cliq2 = np.copy(cliq)
        cliq2[cliq2>=num_ts] = cliq2[cliq2>=num_ts]-num_ts        
        c2 = np.min(cliq2)
        
        if c1==c2:
            count_cliq+= 1
            cliques_gf.append(cliq2.tolist())
            
    return cliques_gf
    
def remove_duplicate_cliqs(cliques_tmp,num_ts):
    cliques_gf = []
    count_cliq = 0
    for i in range(len(cliques_tmp)):
        cliq = np.array(list(cliques_tmp[i]))
        c1 = np.min(cliq)
        
        cliq2 = np.copy(cliq)
        cliq2[cliq2>=num_ts] = cliq2[cliq2>=num_ts]-num_ts        
        c2 = np.min(cliq2)
        
        if c1==c2:
            count_cliq+= 1
            cliques_gf.append(cliq2.tolist())
            
    return cliques_gf
    
def extract_multipole(Cliq,CliqMat,delta):
    sz = np.shape(CliqMat)[0]
    found = False
    CurrMems = range(sz)
    FinalSet = []
    LEV = 0
    LEVG = 0
    
    while (sz>=3):
        CurrMat = CliqMat[np.ix_(CurrMems,CurrMems)]
        W = np.abs(CurrMat) - np.eye(sz)           
        mean_vec = np.divide(np.mean(W,axis=1)*sz,(sz-1)*1.0)
        min_wt = np.min(mean_vec)
        min_ind = np.argmin(mean_vec)
        if min_wt>=delta:
            [LEV,LEVG] = MISC.get_lev_and_levg(CurrMat)
            if LEVG>=delta:    
                found = True
                break
            else:
                LEV = 0
                LEVG = 0 

        del CurrMems[min_ind]
        sz = sz-1
    
    if found:
        FinalSet = [Cliq[i] for i in CurrMems]
    
    return [FinalSet,LEV,LEVG,sz]
    

def find_good_multipoles(cliques_gf,CorrMat,delta):
    AllLEVs = np.zeros([len(cliques_gf),])
    AllLEVGs = np.zeros([len(cliques_gf),])
    AllSizes = np.zeros([len(cliques_gf),])
#    AllMeans = np.zeros([len(cliques),])
    AllMPs = [[] for i in range(len(cliques_gf))]    
    t1 = time.time()
    for i in range(len(cliques_gf)):
        cliq = list(cliques_gf[i])
        AllSizes[i] = len(cliq)
        NewCorrMat = CorrMat[np.ix_(cliq,cliq)]
        [AllMPs[i],AllLEVs[i],AllLEVGs[i],AllSizes[i]] = extract_multipole(cliq,NewCorrMat,delta)
    t2 = time.time()
    print "Time Elapsed:"+str(t2-t1) + " seconds"
    
    GoodInds = np.where(AllLEVGs>0)[0]
    AllMPs2 = [AllMPs[i] for i in GoodInds]
    AllLEVs2 = [AllLEVs[i] for i in GoodInds]
    AllLEVGs2 = AllLEVGs[GoodInds]
    AllSizes2 = AllSizes[GoodInds]
    return [AllMPs2,AllLEVs2,AllLEVGs2,AllSizes2]

def CLIQ_ALGO(CorrMat,delta,edge_filt):
    num_ts = np.shape(CorrMat)[0]
    # Construction of Negative Graph
    CorrMat1 = np.copy(CorrMat)
    CorrMat1[CorrMat1>edge_filt] = 3
    CorrMat1[CorrMat1<=edge_filt] = 1
    CorrMat1[CorrMat1==3] = 0
    
   # Graph1 of Negative Edges
    Graph1 = {}
    for i in range(num_ts):
        Graph1[i] = np.where(CorrMat1[i][:]==1)[0].tolist()
    
    # Find cliques in Graph1
    cliques_g1 = bk.find_cliques(Graph1)  
    
    # Find multipoles with strong linear gain from the obtained cliques
    t1 = time.time()
    [AllMPs2_g1,AllLEVs2_g1,AllLEVGs2_g1,AllSizes2_g1] = find_good_multipoles(cliques_g1,CorrMat,delta)
    t2 = time.time()
    print "Time Elapsed in finding good neg MPs:"+str(t2-t1) + " seconds"

    
    # REMOVE DUPLICATE CLIQUES
    t1 = time.time()
    [FinalMPList_g1,FinalLEVList_g1,FinalLEVGList_g1] = MISC.remove_redundant_multipoles_alter(AllMPs2_g1,AllLEVs2_g1,AllLEVGs2_g1)
    t2 = time.time()
    print "Time Elapsed in eliminating non-maximal/duplicate neg MPs:"+str(t2-t1) + " seconds"

#    FinalSzs_g1 = np.array([len(x) for x in FinalMPList_g1])
#    QuadInds_g1 = np.where(FinalSzs_g1==4)[0].tolist()
#    FinalQuads_g1 = [FinalMPList_g1[i] for i in np.where(FinalSzs_g1==4)[0].tolist()]
#    FinalQuadsLEV_g1 = [FinalLEVList_g1[i] for i in QuadInds_g1]
   
    ##################################
    # FIND PSEUDO-NEGATIVE CLIQUES####
    ##################################
    #CorrMat2: Only retains positive edges
    CorrMat2 = np.copy(CorrMat)
    CorrMat2[CorrMat<-edge_filt] = 0
    CorrMat2[CorrMat>=-edge_filt] = 1
    CorrMat2[CorrMat==1] = 0

    #Graph2: for CorrMat2 
    Graph2 = {} # Only retains positive edges
    for i in range(num_ts):
        Graph2[i] = np.where(CorrMat2[i][:]==1)[0].tolist()
   
    #GraphF: Duplicate copies of Graph1 with all cross-connections being the positive edges
    GraphF = construct_GraphF(Graph1,Graph2,num_ts)

    # Find cliques in GraphF
    t1 = time.time()
    cliques_tmp = bk.find_cliques(GraphF)
    t2 = time.time()
    print "Time Elapsed in generating cliques of GraphF:"+str(t2-t1) + " seconds"
    
    # Remove Negative cliques that are confined to only first half or second half of vertices
    cliques_gf = remove_neg_cliques(cliques_tmp,num_ts)
   
   
    # Find multipoles with strong linear gain from the obtained cliques
    t1 = time.time()
    [AllMPs2,AllLEVs2,AllLEVGs2,AllSizes2] = find_good_multipoles(cliques_gf,CorrMat,delta)
    t2 = time.time()
    print "Time Elapsed in finding good pseudo-neg MPs:"+str(t2-t1) + " seconds"

    
    # REMOVE DUPLICATE MULTIPOLES
    t1 = time.time()
    [FinalMPList_gf,FinalLEVList_gf,FinalLEVGList_gf] = MISC.remove_redundant_multipoles_alter(AllMPs2,AllLEVs2,AllLEVGs2)
    t2 = time.time()
    print "Time Elapsed in eliminating non-maximal/duplicate pseudo-neg MPs:"+str(t2-t1) + " seconds"
    
    # Quads:
#    FinalSzs_gf = np.array([len(x) for x in FinalMPList_gf])
#    QuadInds_gf = np.where(FinalSzs_gf==4)[0].tolist()
#    FinalQuads_gf = [FinalMPList_gf[i] for i in np.where(FinalSzs_gf==4)[0].tolist()]
#    FinalQuadsLEV_gf = [FinalLEVList_gf[i] for i in QuadInds_gf]
    
    FinalMPList = FinalMPList_g1 + FinalMPList_gf
    FinalLEVList = FinalLEVList_g1 + FinalLEVList_gf
    FinalLEVGList = FinalLEVGList_g1 + FinalLEVGList_gf    
    
    return [FinalMPList,FinalLEVList,FinalLEVGList]
    
if __name__ == '__main__':
    sigma = 0.1
    delta = 0.15
    tau = 0.8
    edge_filt = 0
    loaddir = '/panfs/roc/groups/6/kumarv/agraw066/airans/expeditions/Saurabh/QuadPoleAnal/MatlabScripts/'
    data = sio.loadmat(loaddir + 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)+'.mat')
    InputTs = data['FinalTsData']
    InputTs = statm.zscore(InputTs,axis=0)
    CorrMat = np.corrcoef(InputTs,rowvar=0)
    CorrMat = np.nan_to_num(CorrMat)
    num_ts = np.shape(CorrMat)[0]
    
    t_st = time.time()
    [FinalMPList,FinalLEVList,FinalLEVGList] = CLIQ_ALGO(CorrMat,delta,edge_filt)
    t_end = time.time()
    
    print "Total Time for CLIQ FAST: " + str(t_end - t_st) + "seconds"
 

    num_rand = 1000
    [AllStYrs,AllEndYrs,AllWinPvals,AllWinLEVs,AllWinLEVGs] = SIG.pval_anal_SLP(FinalMPList,tau,num_rand)
   
    from os.path import expanduser
    home = expanduser("~")
    savedir = home+'/airans/expeditions/Saurabh/QuadPoleAnal/MultipolesClimate/'
    file_str1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
    file_str2 = 'sigma_'+str(sigma)+'_delta_'+str(delta)   
    
#    loadData = sio.loadmat(savedir+'Multipoles_'+file_str1+file_str2)
#    AllWinPvals = loadData['AllWinPvals']
    saveData = {'FinalMPList':FinalMPList,'FinalLEVList':FinalLEVList,'FinalLEVGList':FinalLEVGList,'AllStYrs':AllStYrs,
                'AllEndYrs':AllEndYrs,'AllWinPvals':AllWinPvals,'AllWinLEVs':AllWinLEVs,'AllWinLEVGs':AllWinLEVGs}
    sio.savemat(savedir+'CLIQUE_Multipoles_'+file_str1+file_str2,saveData,appendmat=True)
    
    num_rand3 = 10**5
    InputSzs = [3,4,5,6]
    [Null3LEVs,Null3LEVGs] = NULL3.pval_anal3_SLP(tau,InputSzs,num_rand3)    
    saveData3 = {'InputSzs':InputSzs,'Null3LEVs':Null3LEVs,'Null3LEVGs':Null3LEVGs}
    sio.savemat(savedir+'NullDistExpt3_'+file_str1+file_str2,saveData3,appendmat=True)