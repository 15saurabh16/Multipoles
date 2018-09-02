# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:51:50 2018

@author: agraw066
"""


import scipy.stats.mstats as statm
import scipy.io as sio
import sys
import numpy.linalg as LA
import time
import pdb
import itertools
import glob

import numpy as np
import os

if __name__ == '__main__':
    dataset = 'SLP' # fMRI
    AllSigma = [0.4]
    AllDelta = [0.1]
    
            
    for s in range(len(AllSigma)):
        sigma = AllSigma[s]
        sigma_str = str(sigma).split('.')[0] + str(sigma).split('.')[1]
        for d in range(len(AllDelta)):
            delta = AllDelta[d]
            delta_str = str(delta).split('.')[0] + str(delta).split('.')[1]                        
            loaddir = os.getcwd() + '/MultipolesLASSOTxtFiles/'+dataset+'/'
            file_name= 'MultipolesLASSO_sigma_'+str(sigma)+'_delta_'+str(delta) #+ '.txt'
            with open(loaddir + file_name) as f:
                AllLines = f.readlines()
                
