# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:37:17 2019

@author: Saurabh
"""

import numpy as np
import sklearn as sk
import scipy.io as sio
import scipy.stats.mstats as statm
import os 
import numpy.linalg as LA
import Misc_Modules as MISC
import pdb
import COMET_ADVANCED as COMETA

def get_multipole_cands(PrecMat,MaxSetSize):
    NewLs = []
    for i in range(PrecMat.shape[0]):
        ids = np.where(np.abs(PrecMat[:,i])>0)[0]
        vals = np.abs(PrecMat[ids,i])
#        vals = vals[0,:]
        sortedinds = np.argsort(vals)[: : -1]
        ids = ids[sortedinds]
        for s in range(3,min(len(sortedinds),MaxSetSize)):    
            NewLs.append(list(ids[:s]))
    return NewLs

def get_lev_and_levgs_all_cands(AllGLMPs,CorrMat):
    AllGLLEVs = [[]]*len(AllGLMPs)
    AllGLLEVGs = [[]]*len(AllGLMPs)
    for k in range(len(AllGLMPs)):
        NewSetCM = CorrMat[np.ix_(AllGLMPs[k],AllGLMPs[k])]
        [lev,levg] = MISC.get_lev_and_levg(NewSetCM)
        AllGLLEVs[k] = lev
        AllGLLEVGs[k] = levg
    return [AllGLLEVs,AllGLLEVGs]


def filter_multipoles(AllGLMPs,AllGLLEVs,AllGLLEVGs,sigma,delta):
    FiltGLMPs = []
    FiltGLLEVs = []
    FiltGLLEVGs = []
    for i in range(len(AllGLMPs)):
        if (1 - AllGLLEVs[i]) >= sigma and AllGLLEVGs[i] >= delta:
            FiltGLMPs.append(AllGLMPs[i])
            FiltGLLEVs.append(AllGLLEVs[i])
            FiltGLLEVGs.append(AllGLLEVGs[i])
    return [FiltGLMPs,FiltGLLEVs,FiltGLLEVGs]


################################################################################
################################################################################    

        
if __name__ == '__main__':
    
    # Parameters
    dataset = 'SLP' #'SLP'
    MaxSetSize = 11
    AllThresh = [0.01,0.1,0.2,0.5,1]
    AllGLMPs = []
    AllSigma = [0.4,0.5,0.6]
    AllDelta = [0.1,0.15,0.2]
    
    loaddir = os.getcwd()

    # Dataset download
    if dataset=='SLP':
        AllAlpha = np.arange(0.1,1,0.01)
        data = sio.loadmat(loaddir + '/psl_NCEP2_C12_1979_2014_73x144_0.8_50_0.8.mat')
        InputTs = data['FinalTsData']
        InputTs = statm.zscore(InputTs,axis=0)
    else:
        AllAlpha = np.arange(0.01,0.4,0.0001)
        State = 'Cartoon'
        scan_id = 7
        data  = sio.loadmat(loaddir+'/AllScanData'+State+'State.mat')
        InputTs = np.transpose(data['AllScansCartoon'][scan_id-1][0])
   
    
    CorrMat = np.corrcoef(InputTs,rowvar=0)
    CorrMat = np.nan_to_num(CorrMat)
    num_ts = np.shape(CorrMat)[0]
    
    for i in range(len(AllAlpha)):
        alpha = AllAlpha[i] # Regularization parameter of Graphical LASSO
        print "alpha ={} ".format(alpha)
        GL = sk.covariance.GraphicalLasso(alpha = alpha)
        GL.fit(InputTs)
        PrecMat = GL.get_precision()
        for j in range(len(AllThresh)):         
            thresh = AllThresh[j] # Threshold on values in Precision matrix to be considered as edges in Markov Network
                        
            # Step 2: Get list of all MPs
            NewLs = get_multipole_cands(PrecMat,MaxSetSize)
            
            # Step 3: Insert it to the end of global list of MPs across all alphas
            AllGLMPs.extend(NewLs)

    
    # Step 4: Calculate LEVs and LEVGs
    print "Computing LEVs and LEVGs"
    [AllGLLEVs,AllGLLEVGs] = get_lev_and_levgs_all_cands(AllGLMPs,CorrMat)
    
    
    print "Now removing non-maximals and duplicates"  
    # Filter the ones with poor sigma and delta and then Remove all non-redundant multipoles
    ParamCombo = []
    FinalResults = []
    for sigma in AllSigma:
        for delta in AllDelta:
            print 'sigma='+str(sigma)+' , delta = '+str(delta)
            ParamCombo.append('sigma='+str(sigma)+' , delta = '+str(delta))
            [FiltGLMPs,FiltGLLEVs,FiltGLLEVGs] = filter_multipoles(AllGLMPs,AllGLLEVs,AllGLLEVGs,sigma,delta)
            [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = COMETA.remove_non_maximals(FiltGLMPs,FiltGLLEVs,FiltGLLEVGs,CorrMat,sigma,delta)
            FinalResults.append([FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList])
            
    
            
    
    
