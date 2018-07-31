# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:05:56 2018

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
import SignifTesterMultipole as SIG
import levg_null_dist_expt3 as NULL
import COMET


def mkdirnotex(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)

    
if __name__ == '__main__':
    sigma = 0.5
    delta = 0.15
    edge_filt = -0.2 # same as mu the parameter of ComEtExtended
    
    loaddir = os.getcwd()
    State = 'Cartoon' # Cartoon
    State2 = 'Rest'
    scan_id = 7
    data  = sio.loadmat(loaddir+'/AllScanData'+State+'State.mat')    
    if State == 'Rest':
        InputTs = np.transpose(data['AllScansRest'][scan_id-1][0] )
    else:
        InputTs = np.transpose(data['AllScansCartoon'][scan_id-1][0])
        
    CorrMat = np.corrcoef(InputTs,rowvar=0)
    CorrMat = np.nan_to_num(CorrMat)
    num_ts = np.shape(CorrMat)[0]
    
    group_sz = 10000
    
    # Calling CoMEtExtended
    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = COMET.COMET_EXT(CorrMat,sigma,delta,edge_filt,group_sz)
    print "Total number of multipoles found for mu = " + str(edge_filt) + " is : "+str(len(FinalMPList)) 
    
    
    
    savedir = os.getcwd() + '/MultipolesfMRI/'
    mkdirnotex(savedir)
    file_str1 = State + '_scanid_'+str(scan_id)
    file_str2 = '_mu_'+str(edge_filt)+'_delta_'+str(delta)+'_sigma_'+str(sigma) 
    
    
    # Analysis of statistical significance and reproducibility of linear gain         
    num_rand = 1000
    [AllScanPvals,AllScanLEVs,AllScanLEVGs] = SIG.pval_anal_fMRI(FinalMPList,State,num_rand)
    [AllScanPvals2,AllScanLEVs2,AllScanLEVGs2] = SIG.pval_anal_fMRI(FinalMPList,State2,num_rand)
    saveData = {'FinalMPList':FinalMPList,'FinalLEVList':FinalLEVList,'FinalLEVGList':FinalLEVGList,'FinalSzList':FinalSzList,
    'AllScanPvals':AllScanPvals,'AllScanPvals2':AllScanPvals2,'AllScanLEVs':AllScanLEVs,'AllScanLEVGs':AllScanLEVGs,
    'AllScanLEVs2':AllScanLEVs2,'AllScanLEVGs2':AllScanLEVGs2}
    sio.savemat(savedir+'CLIQ_COMPLETE_2_Multipoles_'+file_str1+file_str2,saveData,appendmat=True)
    
    
    # Generating Null Distribution of linear dependence of multipoles of different sizes
    num_rand = 10**5
    InputSzs = [3,4,5,6]
    [NullLEVs,NullLEVGs] = NULL.pval_anal3_fMRI(State,InputSzs,num_rand)  
    saveData3 = {'InputSzs':InputSzs,'Null3LEVs':NullLEVs,'Null3LEVGs':NullLEVGs}
    sio.savemat(savedir+'NullDistExpt_'+file_str1+file_str2,saveData3,appendmat=True)