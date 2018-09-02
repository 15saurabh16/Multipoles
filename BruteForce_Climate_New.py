# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 03:10:49 2018

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
#import bronk_kerb as bk
#import SignifTesterMultipole as SIG
#import levg_null_dist_expt3 as NULL3
#import CLIQUE_multipoles_algorithm as CLIQ
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
        
    t2 = time.time()
#    print "Group Time Elapsed:" + str(t2-t1) + " seconds"
    OutputLs = zip(AllMPs,AllLEVs,AllLEVGs)
    return OutputLs
   
#def brute_search_parallel(CorrMat,sz,levgth):
#    AllLEVs = []
#    AllLEVGs = []
#    AllMPs = []
#    num_ts = np.shape(CorrMat)[0]
#    AllInds = range(num_ts)
##    t1 = time.time()
#    AllCombs = []
#    for i in itertools.combinations(AllInds,sz):
#        NewSet = list(i)       
#        AllCombs.append(NewSet)
#    num_proc = 20
#    GroupSz = min(100000,len(AllCombs)/num_proc)
#    GroupSz = max(GroupSz,20000)
#    AllGroups = get_inputs_for_groups(AllCombs,GroupSz)
#    OtherInputs = [CorrMat,levgth]
#    pool = Pool(processes = num_proc)
#    AllGroupOutput = pool.map(brute_search_group, itertools.izip(AllGroups,itertools.repeat(OtherInputs)))                      
#    pool.close() 
#    AllMPsTup,AllLEVsTup,AllLEVGsTup = zip(*sum(AllGroupOutput,[]))
#    AllMPs = list(AllMPsTup)
#    AllLEVs= list(AllLEVsTup)
#    AllLEVGs = list(AllLEVGsTup)
#    return [AllMPs,AllLEVs,AllLEVGs]
    
    
def brute_search_parallel2(CorrMat,sz,sigma,levgth):
    AllLEVs = []
    AllLEVGs = []
    AllMPs = []
    num_ts = np.shape(CorrMat)[0]
    AllInds = range(num_ts)
#    t1 = time.time()
    
    AllCombs = itertools.combinations(AllInds,sz)
    NumCombs = np.divide(math.factorial(num_ts),math.factorial(num_ts-sz)*math.factorial(sz))
    n = 0
    max_chunk_sz = 10**7
    num_proc = 20
    pool = Pool(processes = num_proc)

    while (n<NumCombs-1):
        chunk_sz = min(max_chunk_sz,NumCombs-n-1) #chunk_end - n
        ChnkCombs = []
        for i in range(chunk_sz):           
            NewSet = list(next(AllCombs))       
            ChnkCombs.append(NewSet)
        n = n + chunk_sz 
        print "n = "+str(n)
        GroupSz = min(100000,len(ChnkCombs)/num_proc)
        GroupSz = max(GroupSz,20000)
        print "GroupSz = " +str(GroupSz)
        AllGroups = get_inputs_for_groups(ChnkCombs,GroupSz)
        OtherInputs = [CorrMat,levgth,sigma]
        AllGroupOutput = pool.map(brute_search_group, itertools.izip(AllGroups,itertools.repeat(OtherInputs)))                      
        print "Chunk Finished"
        AllGroupCombined = sum(AllGroupOutput,[])
        if len(AllGroupCombined)>0:            
            ChunkMPsTup,ChunkLEVsTup,ChunkLEVGsTup = zip(*sum(AllGroupOutput,[]))
            AllMPs = AllMPs + list(ChunkMPsTup)
            AllLEVs = AllLEVs + list(ChunkLEVsTup)
            AllLEVGs = AllLEVGs + list(ChunkLEVGsTup)
            
    pool.close()       
    return [AllMPs,AllLEVs,AllLEVGs]


def find_max_edgewt(A):
    sz = A.shape[0]
    V,D = LA.eig(A)           
    levind = np.argmin(V)
    EigVec = D[:,levind]
    NegInds = np.where(EigVec<0)
    A[:,NegInds] = A[:,NegInds]*(-1)
    A[NegInds,:] = A[NegInds,:]*(-1)               
    xs,ys = np.triu_indices(sz,k=1)
    IsNegCl = np.max(A[xs,ys])<=0
    MaxEdgWt = np.max(A[xs,ys])
    return [IsNegCl,MaxEdgWt]

def get_maxedgewt_vec(CorrMat,AllMPs):
    MaxEdgWt = np.zeros((len(AllMPs),))
    for i in range(len(AllMPs)):
        A = CorrMat[np.ix_(AllMPs[i],AllMPs[i])]
        [IsNegCl,MaxEdgWt[i]] = find_max_edgewt(A)
    return [MaxEdgWt]

    
if __name__ == '__main__':
    tau = 0.8
    loaddir = os.getcwd()
    data = sio.loadmat(loaddir + '/psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)+'.mat')
    InputTs = data['FinalTsData']
#    InputTs = np.random.randn(400,1000)
    InputTs = statm.zscore(InputTs,axis=0)
   
    
    AllSigma = [0.6,0.5,0.4]
    AllDelta = [0.2,0.15,0.1]
    sigma = min(AllSigma)
    delta = min(AllDelta)
    t_beg = time.time()
    CorrMat = np.corrcoef(InputTs,rowvar=0)
    CorrMat = np.nan_to_num(CorrMat)
    t1 = time.time()
    [Brut3MPs,Brut3LEVs,Brut3LEVGs] = brute_search_parallel2(CorrMat,3,sigma,delta)
    t2 = time.time()
    print "Time Elapsed for BruteSearch of size 3:"+str(t2-t1)+" seconds"
    print "Total 3Poles: "+str(len(Brut3MPs))
    
    t1 = time.time()
    [Brut4MPs,Brut4LEVs,Brut4LEVGs] = brute_search_parallel2(CorrMat,4,sigma,delta)
    t2 = time.time()
    print "Time Elapsed for BruteSearch of size 4:"+str(t2-t1)+" seconds"
    print "Total 4Poles: "+str(len(Brut4MPs))
    
#    t1 = time.time()
    [Brut5MPs,Brut5LEVs,Brut5LEVGs] = brute_search_parallel2(CorrMat,5,sigma,delta)
    t2 = time.time()
    print "Time Elapsed for BruteSearch of size 5:"+str(t2-t1)+" seconds"
    print "Total 5Poles: "+str(len(Brut5MPs))
    [Brut5MPs,Brut5LEVs,Brut5LEVGs] = [[],[],[]]
    
    
    t1= time.time()
    MaxEdgWt3 = get_maxedgewt_vec(CorrMat,Brut3MPs)
    t2 = time.time()
    print "Time Elapsed for IsNegCliq3:"+str(t2-t1)+" seconds"
    
    t1= time.time()
    MaxEdgWt4 = get_maxedgewt_vec(CorrMat,Brut4MPs)
    t2 = time.time()
    print "Time Elapsed for IsNegCliq4:"+str(t2-t1)+" seconds"
    
    t1= time.time()
    MaxEdgWt5 = get_maxedgewt_vec(CorrMat,Brut5MPs)
    t2 = time.time()
    print "Time Elapsed for IsNegCliq5:"+str(t2-t1)+" seconds"
    
    
    t1= time.time()
    BrutMPs = Brut3MPs + Brut4MPs + Brut5MPs
    BrutLEVs = Brut3LEVs + Brut4LEVs + Brut5LEVs
    BrutLEVGs = Brut3LEVGs + Brut4LEVGs + Brut5LEVGs
#    [BrutMPs,BrutLEVs,BrutLEVGs,BrutSzList]= COMETA.remove_non_maximals(BrutMPs,BrutLEVs,BrutLEVGs,CorrMat,sigma,delta)
#    [BrutMPs,BrutLEVs,BrutLEVGs,BrutSzList] = MISC.remove_redundant_multipoles_alter_parallel(BrutMPs,BrutLEVs,BrutLEVGs)
    t2= time.time()
    print "Time Elapsed i eliminating non-maximals:"+str(t2-t1)+" seconds"
    t_end = time.time()
    TotalTime = t_end - t_beg
    print "Total Time = {}".format(TotalTime)
#    [FinalBrutMPs,FinalBrutLEVs,FinalBrutLEVGs] = MISC.remove_redundant_multipoles_alter(BrutMPs,BrutLEVs,BrutLEVGs)
    NumBrutMPs = []
    ParamCombos = []
   
    for s in range(len(AllSigma)):
        sigma = AllSigma[s]
        for d in range(len(AllDelta)):
            delta = AllDelta[d]
            FinalBrutMPs = [BrutMPs[i] for i in range(len(BrutLEVs))  if (1-BrutLEVs[i]>=sigma and BrutLEVGs[i]>=delta)]
            FinalBrutLEVs = [BrutLEVs[i] for i in range(len(BrutLEVs))  if (1-BrutLEVs[i]>=sigma and BrutLEVGs[i]>=delta)]
            FinalBrutLEVGs = [BrutLEVGs[i] for i in range(len(BrutLEVs))  if (1-BrutLEVs[i]>=sigma and BrutLEVGs[i]>=delta)]
            [FinalBrutMPs,FinalBrutLEVs,FinalBrutLEVGs,FinalBrutSzList]= COMETA.remove_non_maximals(FinalBrutMPs,FinalBrutLEVs,FinalBrutLEVGs,CorrMat,sigma,delta)            
            NumBrutMPs.append(len(FinalBrutMPs))
            ParamCombos.append([sigma,delta])        
            print "sigma = {}, delta = {}, no. of multipoles = {}".format(sigma,delta,len(FinalBrutMPs))
            savedir = os.getcwd()+'/MultipolesClimate/'
            mkdirnotex(savedir)
            file_str1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
            file_str2 = '_sigma_'+str(sigma)+'_delta_'+str(delta)   
            saveData = {'Brut3MPs':Brut3MPs,'Brut3LEVs':Brut3LEVs,'Brut3LEVGs':Brut3LEVGs,
            'Brut4MPs':Brut4MPs,'Brut4LEVs':Brut4LEVs,'Brut4LEVGs':Brut4LEVGs,
            'Brut5MPs':Brut5MPs,'Brut5LEVs':Brut5LEVs,'Brut5LEVGs':Brut5LEVGs,
            'MaxEdgWt3':MaxEdgWt3,'MaxEdgWt4':MaxEdgWt4,'MaxEdgWt5':MaxEdgWt5,
            'FinalBrutMPs':FinalBrutMPs,'FinalBrutLEVs':FinalBrutLEVs,'FinalBrutLEVGs':FinalBrutLEVGs}            
            sio.savemat(savedir+'BRUTE_Multipoles_'+file_str1+file_str2,saveData,appendmat=True)