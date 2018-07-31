# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:40:16 2018

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

def get_sorted_mps(MPList1,LEVList1):
    SortInds1 = np.argsort(np.array(LEVList1))
    SortedMPs1 = []
    for i in range(len(SortInds1)):
        SortedMPs1.append(MPList1[SortInds1[i]])
    SortedLEV1 = np.sort(np.array(LEVList1)).tolist()
    return [SortedMPs1,SortedLEV1,SortInds1]
    
def get_MPs_of_M1_missed_by_M2(MPList1,LEVList1,MPList2,LEVList2):
    [SortedMPs1,SortedLEV1,SortedInds1] = get_sorted_mps(MPList1,LEVList1)
    [SortedMPs2,SortedLEV2,SortedInds2] = get_sorted_mps(MPList2,LEVList2)
    MissedMPs = []
    for i in range(len(SortedMPs1)):
        CurrMP = set(SortedMPs1[i])
        CurrLEV = SortedLEV1[i]
        move_on = False
        j = 0
#        print i
        while (not move_on):
            if j>len(SortedMPs2):
                pdb.set_trace()
                
            CurrMP2 = SortedMPs2[j]
            CurrLEV2 = SortedLEV2[j]
            if CurrLEV2>=CurrLEV or j>=len(SortedMPs2)-1:
                if (not CurrMP.issubset(set(CurrMP2))) or j>=len(SortedMPs2)-1:
                    MissedMPs.append(list(CurrMP))
                    pdb.set_trace()
                move_on = True
            j = j+1
                
    return MissedMPs
                
def get_MPs_of_M1_missed_by_M2_ALTER(MPList1,LEVList1,MPList2,LEVList2):
    [SortedMPs1,SortedLEV1,SortedInds1] = get_sorted_mps(MPList1,LEVList1)
    [SortedMPs2,SortedLEV2,SortedInds2] = get_sorted_mps(MPList2,LEVList2)
    MissedMPs = []
    MissedMPOrigInds = []
    for i in range(len(SortedMPs1)):
        CurrMP = set(SortedMPs1[i])
#        CurrLEV = SortedLEV1[i]
#        print i
        found = False
        for j in range(len(SortedMPs2)):
            CurrMP2 = SortedMPs2[j]
#            CurrLEV2 = SortedLEV2[j]
            if  CurrMP.issubset(set(CurrMP2)):
                found = True
                break
                
        if not found:
            MissedMPs.append(list(CurrMP))
            MissedMPOrigInds.append(SortedInds1[i])
            
    return [MissedMPs,MissedMPOrigInds]             
            
def get_MPs_of_M1_missed_by_M2_NOLEV(MPList1,MPList2):
#    [SortedMPs1,SortedLEV1,SortedInds1] = get_sorted_mps(MPList1,LEVList1)
#    [SortedMPs2,SortedLEV2,SortedInds2] = get_sorted_mps(MPList2,LEVList2)
    
    MissedMPs = []
    MissedMPOrigInds = []
    for i in range(len(MPList1)):
        CurrMP = set(MPList1[i])
#        CurrLEV = SortedLEV1[i]
        print i
        found = False
        for j in range(len(MPList2)):
            CurrMP2 = MPList2[j]
#            CurrLEV2 = SortedLEV2[j]
            if  CurrMP.issubset(set(CurrMP2)):
                found = True
                break
                
        if not found:
            MissedMPs.append(list(CurrMP))
            MissedMPOrigInds.append(i)
            
    return [MissedMPs,MissedMPOrigInds]                      
            
def find_MP_in_MPList(CurrMP,MPList2):
    Ind = -1
    CurrMP = set(CurrMP)
    for j in range(len(MPList2)):
        CurrMP2 = MPList2[j]
#            CurrLEV2 = SortedLEV2[j]
        if  CurrMP.issubset(set(CurrMP2)):
            Ind = j
            break
            
    return Ind