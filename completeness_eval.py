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
#import bronk_kerb as bk
import itertools
import compare_multipole_algorithms as COMP
#import BruteForce_Climate as BRU
import COMET_ADVANCED as COMETA

#AllSzs = [3,4,5]
dataset = 'SLP' # Other option is SLP
AllSigma = [0.4,0.5,0.6]
AllDelta = [0.1,0.15,0.2]
ParamCombo = []
CompletenessCOMET = []
CompletenessLASSO = []

Algo = 'COMET'

for s in range(len(AllSigma)):
    sigma = AllSigma[s]
    for d in range(len(AllDelta)):
        delta = AllDelta[d]
        
        if dataset=='SLP': 
            # Load Data
            tau = 0.8
            data = sio.loadmat(os.getcwd() + '/psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)+'.mat')
            InputTs = data['FinalTsData']
            InputTs = statm.zscore(InputTs,axis=0)
            CorrMat = np.corrcoef(InputTs,rowvar=0)
            CorrMat = np.nan_to_num(CorrMat)
            
            # Load Multipoles of COMET 
            tau = 0.8
            loaddir = os.getcwd()+'/MultipolesClimate/'
            if sigma==0.6 and delta==0.1:            
                edge_filt = 0.03
            else:
                edge_filt = 0.01
            
            file_str1 = 'psl_NCEP2_C12_1979_2014_73x144_0.8_50_'+str(tau)
            file_str2 = '_mu_'+str(edge_filt)+'_sigma_'+str(sigma)+'_delta_'+str(delta) 
            Output1 = sio.loadmat(loaddir + 'COMET_Multipoles_'+file_str1+file_str2+'.mat')
            
            # Load Miltipoles of Brute Force            
            file_strb2 = '_sigma_'+str(sigma)+'_delta_'+str(delta) 
            file_prfx = 'BRUTE_Multipoles_'
            Output_br = sio.loadmat(loaddir+file_prfx+file_str1+file_strb2+'.mat')

            # Load Multipoles of LASSOMultipoles
            file_prfx = 'LASSOMultipoles_'
            Output_bse = sio.loadmat(loaddir+file_prfx+file_str1+file_strb2+'.mat')

            file_prfx = 'RANDSRCH_Multipoles_'
            Output_rnd = sio.loadmat(loaddir+file_prfx+file_str1+file_strb2+'.mat')
            
        elif dataset=='fMRI':            
            #Load Data
            State = 'Cartoon' # Cartoon
            State2 = 'Rest'
            scan_id = 7
            data  = sio.loadmat(os.getcwd()+'/AllScanData'+State+'State.mat')    
            if State == 'Rest':
                InputTs = np.transpose(data['AllScansRest'][scan_id-1][0] )
            else:
                InputTs = np.transpose(data['AllScansCartoon'][scan_id-1][0])
                
            CorrMat = np.corrcoef(InputTs,rowvar=0)
            CorrMat = np.nan_to_num(CorrMat)
            
            # Load Multipoles of COMET 
            loaddir = os.getcwd() + '/MultipolesfMRI/'
            scan_id = 7
            statevar = 'Cartoon'
            if sigma==0.6 and delta==0.1:            
                edge_filt = 0.25
            else:
                edge_filt = 0.2
            file_str1 = '_'+statevar+'_scanid_'+str(scan_id)
            file_str2 = '_mu_'+str(edge_filt)+'_sigma_'+str(sigma)+'_delta_'+str(delta)    
            Output1 = sio.loadmat(loaddir + 'COMET_MultipolesfMRI'+file_str1+file_str2+'.mat')
        
            # Load Multipoles of Brute Force
            loaddir_b =  os.getcwd() + '/MultipolesfMRI/'
            file_strb1 = statevar+'_scanid_'+str(scan_id)
            file_strb2 = '_sigma_'+str(sigma)+'_delta_'+str(delta)    
            Output_br = sio.loadmat(loaddir_b+'BRUTE_Multipoles_'+file_strb1+file_strb2+'.mat')
            
            # Load Multipoles of LASSO
            loaddir_bse =  os.getcwd() + '/MultipolesfMRI/'
            file_strbse1 = statevar+'_scanid_'+str(scan_id)
            file_strbse2 = '_sigma_'+str(sigma)+'_delta_'+str(delta)
            Output_bse = sio.loadmat(loaddir_bse+'LASSOMultipoles_'+file_strbse1+file_strbse2+'.mat')

            # Load Multipoles of Random Search
            file_prfx = 'RANDSRCH_Multipoles'
            Output_rnd = sio.loadmat(loaddir+file_prfx+file_str1+file_strb2+'.mat')

        FinalMPList1 = list(Output1['FinalMPList'][0])
        FinalMPListNew = [FinalMPList1[i].tolist()[0] for i in range(len(FinalMPList1))]
        FinalMPList1 = FinalMPListNew[:]
        FinalLEVList1= list(Output1['FinalLEVList'][0])
        FinalLEVGList1 = list(Output1['FinalLEVGList'][0])
    
        FinalBrutMPs = list(Output_br['FinalBrutMPs'][0])
        FinalMPListNew = [FinalBrutMPs[i].tolist()[0] for i in range(len(FinalBrutMPs))]
        FinalBrutMPs = FinalMPListNew[:]
        FinalBrutLEVs= list(Output_br['FinalBrutLEVs'][0])
        FinalBrutLEVGs = list(Output_br['FinalBrutLEVGs'][0])
        
        if len(Output_bse['FinalBseMPs'])>0:        
            FinalBseMPs = list(Output_bse['FinalBseMPs'])#list(Output_bse['FinalBseMPs'][0])
            FinalMPListNew = [FinalBseMPs[i].tolist() for i in range(len(FinalBseMPs))]
            FinalBseMPs = FinalMPListNew[:]
            FinalBseLEVs= list(Output_bse['FinalBseLEVs'][0])
            FinalBseLEVGs = list(Output_bse['FinalBseLEVGs'][0])
        else:
            FinalBseMPs,FinalBseLEVs,FinalBseLEVGs = [],[],[]
        
        if len(Output_rnd['FinalRndMPs'])>0:        
            FinalRndMPs = list(Output_rnd['FinalRndMPs'][0])
            FinalMPListNew = [FinalRndMPs[i].tolist()[0] for i in range(len(FinalRndMPs))]
            FinalRndMPs = FinalMPListNew[:]
            FinalRndLEVs= list(Output_rnd['FinalRndLEVs'][0])
            FinalRndLEVGs = list(Output_rnd['FinalRndLEVGs'][0])
        else:
            FinalRndMPs,FinalRndLEVs,FinalRndLEVGs = [],[],[]

        print "HELLO"
        # Generate Pseudo-set
        AllMPs = FinalBrutMPs + FinalMPList1 + FinalBseMPs + FinalRndMPs
        AllLEVs = FinalBrutLEVs + FinalLEVList1 + FinalBseLEVs + FinalRndLEVs
        AlllEVGs = FinalBrutLEVGs + FinalLEVGList1 + FinalBrutLEVGs + FinalRndLEVGs
        
        [FinalSetMPs,FinalSetLEVs,FinalSetLEVGs,_] = COMETA.remove_non_maximals(AllMPs,AllLEVs,AlllEVGs,CorrMat,sigma,delta)
        
        [MissedMPsCOMET,_] = COMP.get_MPs_of_M1_missed_by_M2_ALTER(FinalSetMPs,FinalSetLEVs,FinalMPList1,FinalLEVList1)       
        TotalMissMPFracCOMET = np.divide(len(MissedMPsCOMET),float(len(FinalSetMPs))) 

        [MissedMPsLASSO,_] = COMP.get_MPs_of_M1_missed_by_M2_ALTER(FinalSetMPs,FinalSetLEVs,FinalBseMPs,FinalBseLEVs)               
        TotalMissMPFracLASSO = np.divide(len(MissedMPsLASSO),float(len(FinalSetMPs)))
        
        print "Total Completeness at sigma = {}, delta = {}  is {} ".format(sigma, delta, 1-TotalMissMPFracCOMET)
        CompletenessCOMET.append(round(1-TotalMissMPFracCOMET,2))
        CompletenessLASSO.append((1-TotalMissMPFracLASSO))
        ParamCombo.append([sigma,delta])
        #[MissMaxEdgWtVec] = BRU.get_maxedgewt_vec(CorrMat,MissedMPs) 
        #BadMissInds =  np.where(np.array(MissMaxEdgWtVec)<0)[0]