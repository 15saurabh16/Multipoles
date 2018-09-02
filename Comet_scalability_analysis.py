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
#import bronk_kerb as bk
#import SignifTesterMultipole as SIG
#import levg_nu1ll_dist_expt3 as NULL3
#import itertools
#import CLIQ_COMPLETE as CLIQC_OLD # Wrong Graph Construction 
import COMET_ADVANCED as COMETA 

def mkdirnotex(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)


if __name__ == '__main__':
    tau = 0.8
    AllSigma = [0.4]
    AllDelta = [0.1]
    AllMu = [-0.06]#[-0.09] #[-0.15,-0.12,-0.09,-0.06]
    for s in range(len(AllSigma)):
        sigma = AllSigma[s] # 0.5 0.4 0.6
        for d in range(len(AllDelta)):
            delta = AllDelta[d]
            if (sigma==0.5 and delta==0.15): # BECAUSE ALREADY DONE
                continue
            
            for e in range(len(AllMu)):
                edge_filt = AllMu[e]
                
                if (edge_filt== -0.06):
                    max_seas = 4
                else:
                    max_seas = 10

                AllTimeComet = np.zeros((max_seas,))
                
                for i in range(max_seas):        
                    num_seas = i+1
                    loadfile = 'psl_NCEP2_1979_2014_73x144_50_'+str(tau)+'_'+str(num_seas)+'_seasons.mat'
                    loaddir = os.getcwd()  + '/ScalablityData/'
                    data = sio.loadmat(loaddir + loadfile)
                    InputTs = data['CurrDataset']
                    Graph_prefix = 'GraphFSLP'
                    GraphStr = Graph_prefix+str(num_seas)+str(edge_filt)
                    
                    t1 = time.time()
                    InputTs = statm.zscore(InputTs,axis=0)
                    CorrMat = np.corrcoef(InputTs,rowvar=0)
                    CorrMat = np.nan_to_num(CorrMat)
                    num_ts = np.shape(CorrMat)[0]
                    
                    group_sz = 1000
                    [FinalMPList,FinalLEVList,FinalLEVGList,FinalSzList] = COMETA.COMET_EXT(CorrMat,sigma,delta,edge_filt,group_sz,GraphStr)
                    print "Total number of multipoles found for mu = " + str(edge_filt) + " is : "+str(len(FinalMPList)) 
                    t2 = time.time()
                    AllTimeComet[i] = t2-t1
                    print "sigma: {}, delta: {}, mu: {}, AllTimeComet[{}] = {}".format(sigma,delta,edge_filt,i,AllTimeComet[i])
                
          
                savedir = os.getcwd()+'/ScalabilityResults/'
                mkdirnotex(savedir)
                file_str1 = 'psl_'
                file_str2 = '_mu_'+str(edge_filt)+'_sigma_'+str(sigma)+'_delta_'+str(delta)    
                file_str3 = '_num_seasons_'+str(max_seas)                                        
                saveData = {'AllTimeComet':AllTimeComet}               
                sio.savemat(savedir+'Scalability_Multipoles_'+file_str1+file_str2+file_str3,saveData,appendmat=True)
