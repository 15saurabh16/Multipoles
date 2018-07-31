# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:19:39 2018

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
#import CLIQ_COMPLETE as CLIQC_OLD # Wrong Graph Construction 
import COMET 

def mkdirnotex(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)


if __name__ == '__main__':
    sigma = 0.1
    delta = 0.15
    tau = 0.8
    edge_filt = 0
    loaddir = os.getcwd()
    data = sio.loadmat(loaddir + '/psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)+'.mat')
    InputTs = data['FinalTsData']
    InputTs = statm.zscore(InputTs,axis=0)
    CorrMat = np.corrcoef(InputTs,rowvar=0)
    CorrMat = np.nan_to_num(CorrMat)
    num_ts = np.shape(CorrMat)[0]
    
    group_sz = 10000
    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = COMET.COMET_EXT(CorrMat,sigma,delta,edge_filt,group_sz)
    print "Total number of multipoles found for mu = " + str(edge_filt) + " is : "+str(len(FinalMPList)) 

#    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = CLIQC.CLIQ_COMPLETE(CorrMat,delta,edge_filt)
#    print "Total number of multipoles found for mu = " + str(edge_filt) + " is : "+str(len(FinalMPList)) 
#    [FinalMPList1,FinalLEVList1,FinalLEVGList1,FinalSzList1] = CLIQC_OLD.CLIQ_COMPLETE(CorrMat,delta,edge_filt)
#    print "Total number of multipoles found for mu = " + str(edge_filt) + " is : "+str(len(FinalMPList)) 
#    pdb.set_trace()
       
    from os.path import expanduser
    home = expanduser("~")
    savedir = os.getcwd()+'/MultipolesClimate/'
    mkdirnotex(savedir)
    file_str1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
    file_str2 = '_mu_'+str(edge_filt)+'_sigma_'+str(sigma)+'_delta_'+str(delta)    
    
#    pdb.set_trace()
    
    if edge_filt==0:  
        num_rand = 1000
        [AllStYrs,AllEndYrs,AllWinPvals,AllWinLEVs,AllWinLEVGs] = SIG.pval_anal_SLP(FinalMPList,tau,num_rand)
    else:
        AllStYrs = []
        AllEndYrs = []
        AllWinPvals = []
        AllWinLEVs = []
        AllWinLEVGs = []
        
        
    saveData = {'FinalMPList':FinalMPList,'FinalLEVList':FinalLEVList,'FinalLEVGList':FinalLEVGList,'FinalSzList':FinalSzList,
    'AllStYrs':AllStYrs,'AllEndYrs':AllEndYrs,'AllWinPvals':AllWinPvals,'AllWinLEVs':AllWinLEVs,'AllWinLEVGs':AllWinLEVGs}
    sio.savemat(savedir+'CLIQ_COMPLETE_2_Multipoles_'+file_str1+file_str2,saveData,appendmat=True)

    num_rand3 = 10**5
    InputSzs = [3,4,5,6]
    [Null3LEVs,Null3LEVGs] = NULL3.pval_anal3_SLP(tau,InputSzs,num_rand3)    
    saveData3 = {'InputSzs':InputSzs,'Null3LEVs':Null3LEVs,'Null3LEVGs':Null3LEVGs}
    sio.savemat(savedir+'NullDistExpt3_'+file_str1+file_str2,saveData3,appendmat=True)