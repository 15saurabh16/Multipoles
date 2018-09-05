# Multipoles
This repository contains the code that we used to find "multipoles", a new class of multivariate relationship patterns in time series data in datasets from climate and neuroscience domain. See [1] for further details. 

<b>Instructions to run the code:</b>

1) The code to obtain all the empirical observations discussed in section 4.1 of techincal report can be obtained using empirical_observations.py.


2) Please keep all the codes and data files in the same directory.
3) There are two implementations of COMET algorithm; COMET.py and COMET_ADVANCED.py. The two implementations mainly differ in the choice of algorithm used to solve the clique enumeration problem in one of the intermediate steps. Script COMET.py uses bronk_kerb.py, a python implementation of bronk kerbosch's algorithm (1973), whereas COMET_ADVANCED.py uses the C++ implementation (in quick-cliques folder) provided by authors (available at https://github.com/darrenstrash/quick-cliques).   contains the primary module COMET_EXT which can be called to find all multipoles in a given dataset. COMET_EXT takes five inputs:<br>
	CorrMat: Correlation matrix of the entire dataset <br>
	sigma: minimum threshold on linear dependence of desired multipoles
	delta: minimum threshold on linear gain of desired multipoles
	edge_filt: same as parameter \mu of CoMEtExtended (see <a href = "https://www.researchgate.net/publication/323129038_Mining_Novel_Multivariate_Relationships_in_Time_Series_Data_Applications_to_Climate_and_Neuroscience"> technical report </a> for further details). Higher values of edge_filt will recover more multipoles but will increase computational time and vice-versa.   
	group_sz: This code runs in parallel on 20 processors. Each processor examines a group of candidates to find multipoles. group_sz is the desired size of a group. I set it to 10000 on my machine.
  Graph_str(only in the version of COMET_ADVANCED.py): A filename to be given to the text files that store the correlation graphs that are used to find cliques.  
  In the output, four lists are obtained: i) List of all multipoles, ii) List of their corresponding least eigenvalues (= 1 - linear dependence), iii) List of corresponding linear gains, and iv) LIst of their corresponding sizes. 

4) Script COMET_fMRI.py was used to find all multipoles in fMRI data (using COMET.py) and also computes statistical significance of all the obtained multipoles. 
5) Script COMET_Climate.py was used to find all multipoles in SLP data (using COMET.py) and also computes statistical significance of all the obtained multipoles.
6) Script COMET_Climate_Multiple_Parameters.py uses COMET_ADVANCED.py to find all multipoles in SLP data (using COMET_ADVANCED.py)
7) Script COMET_Climate_Multiple_Parameters.py uses COMET_ADVANCED.py to find all multipoles in fMRI data (using COMET_ADVANCED.py)
8) BruteForce_fMRI.py finds all multipoles in fMRI data using a brute-force approach. 
9) Script BruteForce_Climate.py finds all multipoles in SLP data using a brute-force approach. 
10) Script compare_with_brute_force.py can be used to compare the output of COMET_EXT with brute force.

# References: 
[1] <a href = "https://www.researchgate.net/publication/323129038_Mining_Novel_Multivariate_Relationships_in_Time_Series_Data_Applications_to_Climate_and_Neuroscience"> Technical Report </a>
