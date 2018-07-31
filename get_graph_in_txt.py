# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:49:34 2018

@author: agraw066
"""

import pdb
import pickle
import os
def get_graph_in_txt(GraphStr,GraphTxtFile):
    #'/panfs/roc/groups/6/kumarv/agraw066/airans/expeditions/Saurabh/QuadPoleAnal/GraphTxtFiles/'
    with open(GraphStr+'.pkl') as f:
        [GraphF]= pickle.load(f)


   
    N = len(GraphF)
    
    # Get number of edges
    E = 0
    for i in range(N):
        E += len(GraphF[i])
    
    with open(GraphTxtFile,'w') as f:
        f.write('{} {}\n'.format(N,E))
    
    f = open(GraphTxtFile,'a')
    edges =0
    for i in range(N):
        v1 = i
        for j in range(len(GraphF[v1])):
            edges += 1
            v2  = GraphF[v1][j]
            f.write('{},{}\n'.format(v1,v2))
    
    f.close()
    
    return


if __name__ == '__main__':
    num_seas = 4
    edge_filt = -0.01#0 #-0.1
    GraphPref = 'GraphFfMRI'#'GraphFSLP' # 'GraphFRand'
    GraphStr = GraphPref+str(num_seas)+str(edge_filt)
    with open(GraphStr+'.pkl') as f:
        [GraphF]= pickle.load(f)

   # fUNCTION BODY
    savedir = os.getcwd() + '/GraphTxtFiles/'
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    savefile = GraphStr+'.txt'
    
    N = len(GraphF)
    
    # Get number of edges
    E = 0
    for i in range(N):
        E += len(GraphF[i])
    
    with open(savedir+savefile,'w') as f:
        f.write('{} {}\n'.format(N,E))
    
    f = open(savedir+savefile,'a')
    edges =0
    for i in range(N):
        v1 = i
        for j in range(len(GraphF[v1])):
            edges += 1
            v2  = GraphF[v1][j]
            f.write('{},{}\n'.format(v1,v2))
    
    f.close()
    
    
    np.random.randn()