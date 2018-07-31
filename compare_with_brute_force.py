# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:56:38 2018

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
import compare_multipole_algorithms as COMP
import BruteForce_Climate as BRU

#AllSzs = [3,4,5]
dataset = 'SLP' # Other option is SLP
AllSigma = [0.4,0.5,0.6]
AllDelta = [0.1,0.15,0.2]
ParamCombo = []
Completeness = []

for s in range(len(AllSigma)):
    sigma = AllSigma[s]
    for d in range(len(AllDelta)):
        delta = AllDelta[d]
        
        if dataset=='SLP':   
            tau = 0.8
            loaddir = os.getcwd()+'/MultipolesClimate/'
            edge_filt = 0.01
            file_str1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
            file_str2 = '_mu_'+str(edge_filt)+'_sigma_'+str(sigma)+'_delta_'+str(delta) 
            Output1 = sio.loadmat(loaddir + 'COMET_Multipoles_'+file_str1+file_str2+'.mat')
            
            loaddir = os.getcwd()
            data = sio.loadmat(loaddir + '/psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)+'.mat')
            InputTs = data['FinalTsData']
            InputTs = statm.zscore(InputTs,axis=0)
            CorrMat = np.corrcoef(InputTs,rowvar=0)
            CorrMat = np.nan_to_num(CorrMat)
            
        elif dataset=='fMRI':
            loaddir = os.getcwd() + '/MultipolesfMRI/'
            scan_id = 7
            statevar = 'Cartoon'
            edge_filt = 0.2
            file_str1 = '_'+statevar+'_scanid_'+str(scan_id)
            file_str2 = '_mu_'+str(edge_filt)+'_sigma_'+str(sigma)+'_delta_'+str(delta)    
            Output1 = sio.loadmat(loaddir + 'COMET_MultipolesfMRI'+file_str1+file_str2+'.mat')
        
        FinalMPList1 = list(Output1['FinalMPList'][0])
        FinalMPListNew = [FinalMPList1[i].tolist()[0] for i in range(len(FinalMPList1))]
        FinalMPList1 = FinalMPListNew[:]
        FinalLEVList1= list(Output1['FinalLEVList'][0])
        FinalLEVGList1 = list(Output1['FinalLEVGList'][0])
        
        from os.path import expanduser
        home = expanduser("~")
        
        if dataset=='SLP':
            loaddir_b =  os.getcwd() + '/MultipolesClimate/'
            file_strb1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
            file_strb2 = '_sigma_'+str(sigma)+'_delta_'+str(delta) 
            Output_br = sio.loadmat(loaddir_b+'BRUTE_Multipoles_'+file_strb1+file_strb2+'.mat')
            
        elif dataset=='fMRI':
            loaddir_b =  os.getcwd() + '/MultipolesfMRI/'
            file_strb1 = statevar+'_scanid_'+str(scan_id)
            file_strb2 = '_sigma_'+str(sigma)+'_delta_'+str(delta)    
            Output_br = sio.loadmat(loaddir_b+'BRUTE_Multipoles_'+file_strb1+file_strb2+'.mat')
        
        
        FinalBrutMPs = list(Output_br['FinalBrutMPs'][0])
        FinalMPListNew = [FinalBrutMPs[i].tolist()[0] for i in range(len(FinalBrutMPs))]
        FinalBrutMPs = FinalMPListNew[:]
        FinalBrutLEVs= list(Output_br['FinalBrutLEVs'][0])
        FinalBrutLEVGs = list(Output_br['FinalBrutLEVGs'][0])
        
        
        [MissedMPs,MissedMPOrigInds] = COMP.get_MPs_of_M1_missed_by_M2_ALTER(FinalBrutMPs,FinalBrutLEVs,FinalMPList1,FinalLEVList1)
        [MissedMPs2,MissedMPOrigInds2] = COMP.get_MPs_of_M1_missed_by_M2_ALTER(FinalMPList1,FinalLEVList1,FinalBrutMPs,FinalBrutLEVs)
        #MissedMPFracSzs = np.zeros([len(AllSzs),])
        #for i in range(len(AllSzs)):
        #    sz = AllSzs[i]
        #    RelInds = [j for j in range(len(FinalBrutMPs)) if (len(FinalBrutMPs[j])==sz)]
        #    NumMPsSz = len(RelInds)
        #    RelInds2 = [j for j in range(len(MissedMPs)) if (len(MissedMPs[j])==sz)] 
        #    NumMissMPsSz = len(RelInds2)
        #    MissedMPFracSzs[i] = np.divide(NumMissMPsSz,float(NumMPsSz))
        #    print "Complteness for size "+str(sz)+ " = "+str(1-MissedMPFracSzs[i])
        
        TotalMissMPFrac = np.divide(len(MissedMPs),float(len(FinalBrutMPs)+len(MissedMPs2))) 
        print "Total Completeness at sigma = {}, delta = {}  is {} ".format(sigma, delta, 1-TotalMissMPFrac)
        Completeness.append(round(1-TotalMissMPFrac,2))
        ParamCombo.append([sigma,delta])
        #[MissMaxEdgWtVec] = BRU.get_maxedgewt_vec(CorrMat,MissedMPs) 
        #BadMissInds =  np.where(np.array(MissMaxEdgWtVec)<0)[0]