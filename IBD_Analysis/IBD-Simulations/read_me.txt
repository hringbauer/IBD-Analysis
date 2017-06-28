This is the read-me for the SpatialBlockSim part of the project. It generates and partly analyzes block sharing data from node models. Everything is written in Python - it relies on packages such as numpy, scipy and matplotlib. Make sure to install all of them (for instance via PIP)

There are several classes/files dividing tasks among them:

Main: This is the file where everything comes together; and it provides the text-interface for the user in a do your own adventure style. The user inputs integers to navigate trough the menues, in a do-your-own adventure type menu.

Grid: This class contains the method to simulate the data. IT ALSO CONTAINS the parameters for which grid to run. In addition to the constant population size Grid object, it contains a factory method to give back a Grid_Grow class, which can be used to simulate data of a growing or declining population - with parameters specified THERE.

Parent_Draw: Class that draws the relative position of the parents. For a deme model, also contains the parameters for which deme size to use.

blockpiece: Contains a class describing single blocks (BlPiece) and also one that can describe multiple blocks (Multi_Bl). This is used for model pieces on a single node of the Grid object




############################################################################################################################
User-Manual:

Start running main.py:

1) Set blocks: This sets the initial IBD-blocks as specified by the grid.py file
2) Update Generation: This runs the simulation for a specified number of generations.
3) Data Analaysis: Now you can do several analysis tasks  OR
4) Full MLE-analysis: This loads the mle_analysis object from POPRES analysis, that allows to one to do maximum likelihood analysis. When doing so, first the block sharing data is loaded into the appropriate frame work (list of distances, list of pairwise shared blocks; and if wanted - data binned with respect to pairwise distance - as on a grid often multiple individuals will have the same such distance)

There one first has to choose the scenario to fit via "Choose MLE-model". Then a single run can be done via "Run Fit". This will yield the MLE-parameters and some statistics. A summary graphical fit can then be done via "Log-Likelihood surface"; or the residual plot done vie "Analyze Residuals"

One can do bootstrap over different units; and summary statistics are printed out then. To visualize the bootstrap results; one can plot the Log-Likelihood surface.
 

