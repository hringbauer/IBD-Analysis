This is the read-me for the DISC-SIM part of the project. It generates and partly analyzes block sharing data from the DISCSIM  models. Everything is written in Python - it relies on packages such as numpy, scipy and matplotlib. Make sure to install all of them.(for instance via PIP) Crucially, it uses DISCSIM as implemented by Jerome Kelleher, which can be found at (for instance via PIP)

There are several classes/files dividing tasks among them:

Main: This is the file where everything comes together; and it provides the text-interface for the user in a do your own adventure style. The user inputs integers to navigate trough the menues, in a do-your-own adventure type menu. This also contains the parameters to which data is simulated (u, r, num_loci etc.)

IBD-detection: This is the part where actually the IBD-detection happens, and contains parameters for this. It analyzes the coalescent tree output by DISCSIM (tau and pi) and scans for loci where the coalescence tree jumps. It can be configured to count all jumps; or only the ones which switch between ancestral and non-ancestral time (given by parameter), to account for the effect of non-effective recombination events.

It also imports from the MLE_analyse class from the POPRES analysis, to have the same code analyzing the simulated and the empirical data set.

units: Contains some methods to transform units such that they are per Generation in DISCSIM.

multi_runs: This file contains software to do multiple runs for various visualization purpose. For this it directly loads the necessary code; and partly
from the mle_analysis object from the POPRES analysis (to have the same code analyzing simulated and empirical data). This has it's own menue, where one can create data sets; which are usually saved with pickle and then used for analysis.




############################################################################################################################
User-Manual:

Start running main.py:

1) Run DISCSIM
2) Detect IBD-blocks; either for effective or all recombination events
3) Do usual analysis of IBD-blocks; or analysis with the MLE-analysis object (see there)
4) One can save/load data with pickle

For multiple runs, run multi_runs directly.
