# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 22:09:41 2018

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
import COMET_ADVANCED as COMETA # Uses tomita algorithm to find cliques
#import CLIQ_COMPLETE as CLIQC_OLD # Wrong Graph Construction 
#import COMET # Uses basic bron-kerbosch algorithm to find cliques

def mkdirnotex(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)


if __name__ == '__main__':
    AllSigma = [0.4,0.5,0.6]
    AllDelta = [0.1,0.15,0.2]
    AllMu = [-0.2,-0.1,0,0.1,0.2]
   
    
    loaddir = os.getcwd()
    State = 'Cartoon' # Cartoon
    State2 = 'Rest'
    scan_id = 7
    data  = sio.loadmat(loaddir+'/AllScanData'+State+'State.mat')    
    if State == 'Rest':
        InputTs = np.transpose(data['AllScansRest'][scan_id-1][0] )
    else:
        InputTs = np.transpose(data['AllScansCartoon'][scan_id-1][0])

    ParamCombo = []
    NumMultipoles = []
    TotalTime = []
    
    for s in range(len(AllSigma)):
        sigma = AllSigma[s]
        for d in range(len(AllDelta)):
            delta = AllDelta[d] 
            for e in range(len(AllMu)):
                
                
                edge_filt = AllMu[e]
                t_beg = time.time()
                CorrMat = np.corrcoef(InputTs,rowvar=0)
                CorrMat = np.nan_to_num(CorrMat)
                num_ts = np.shape(CorrMat)[0]

                group_sz = 10000
                GraphStr = 'GraphFfMRI_'+str(sigma) + '_' +str(delta) + '_' + str(edge_filt)
                [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = COMETA.COMET_EXT(CorrMat,sigma,delta,edge_filt,group_sz,GraphStr)
                t_end = time.time()
                ParamCombo.append([sigma,delta,edge_filt])
                TotalTime.append(t_end-t_beg)
                NumMultipoles.append(len(FinalMPList))
                
                savedir = os.getcwd()+'/MultipolesfMRI/'
                mkdirnotex(savedir)
                file_str1 = State + '_scanid_'+str(scan_id)
                file_str2 = '_mu_'+str(edge_filt)+'_sigma_'+str(sigma)+'_delta_'+str(delta)                                                        
                saveData = {'FinalMPList':FinalMPList,'FinalLEVList':FinalLEVList,'FinalLEVGList':FinalLEVGList,'FinalSzList':FinalSzList}
                sio.savemat(savedir+'COMET_MultipolesfMRI_'+file_str1+file_str2,saveData,appendmat=True)
                print "sigma:{}, delta:{}, mu:{}, NumMultipoles:{}, TotalTime:{}".format(sigma,delta,edge_filt,len(FinalMPList),t_end-t_beg)

    