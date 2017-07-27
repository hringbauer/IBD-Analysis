'''
Created on Feb 18, 2016

@Harald: Contains class for MLE estimaton with
estimated error. Everything here is measured in cM
Most quantities are for probabilities per pair
'''
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import kv as kv  # Import Bessel functions of second kind
from bisect import bisect_left, bisect_right
from time import time
from hetero_sharing import ibd_sharing, prepare_step_size, centering_positions
import matplotlib.pyplot as plt
import numpy as np
    
class MLE_estim_error(GenericLikelihoodModel):
    '''
    Class for MLE estimation of block sharing
    between populations with modeled error
    Bins into length blocks and models block-sharing
    between populations as indep. Poisson
    '''
    # Bins for mle_multi_run
    min_b, max_b = 0, 30.1  # Minimum/Maximum bin for mle_multi_run    # 30.2
    bin_width = 0.1  # Bin width for mle_multi_run
    min_len, max_len = 4.0, 20.0  # Minimum/maximum bin length actually analyzed    #20
    min_ind, max_ind = 0, 0  # Indices for start stop of bins of interest
    mid_bins = []  # Array for the bins
    
    fp_rate = []  # Array for false positives
    theoretical_shr = []  # Array for theoretical expected sharing per bin
    trans_mat = np.zeros((2, 2))  # Transition matrix for theor. to expected block-sharing
    full_shr_pr = []  # Array for the full bin sharing 
      
    density_fun = 0  # function used to calculate the block sharing density; is required to be per cM!!
    start_params = []  # List of parameters for the starting array
    error_model = True  # Parameter whether to use error model
    estimates = []  # The last parameter which has been fit
    
    def __init__(self, bl_dens_fun, start_params, pw_dist, pw_IBD, pw_nr, error_model=True, **kwds):
        '''Takes the function; start parameters and three important lists as input:
        List of pw. distances, list of pw. nr and list of pw. IBD-Lists (in cM)'''
        exog = np.column_stack((pw_dist, pw_nr))  # Stack the exogenous variables together
        endog = pw_IBD
        self.initialize_ll_model(endog, exog)
        self.create_bins()  # Create the Mid Bin vector
        self.fp_rate = fp_rate(self.mid_bins) * self.bin_width  # Calculate the false positives per bin
        self.density_fun = bl_dens_fun  # Set the block density function 
        self.start_params = start_params 
        self.error_model = error_model  # Whether to use error model
        if self.error_model == True:  # In case required:  
            self.calculate_trans_mat()  # Calculate the Transformation matrix
    
    def initialize_ll_model(self, endog, exog, **kwds):
        '''Function to initialize LL model'''
        super(MLE_estim_error, self).__init__(endog, exog, **kwds)  # Create the full object.
        
    def loglikeobs(self, params):
        '''Return vector of log likelihoods for every observation. (here pairs of pops)'''
        for i in range(len(params)):
            print("Parameter %.0f : %.8f" % (i, params[i]))
        C = params[0]  # Absolute Parameter
        sigma = params[1]  # Dispersal parameter

        if C <= 0 or sigma <= 0:  # If Parameters do not make sense return infinitely negative likelihood
            return -np.ones(len(self.endog)) * (np.inf)
        
        ll = [self.pairwise_ll(self.endog[i], self.exog[i, :], params) for i in range(len(self.endog))]
        print("Total log likelihood: %.4f" % np.sum(ll))
        return np.array(ll).astype('float')  # Return negative log likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = self.start_params  # Set the starting parameters for the fit
        fit = super(MLE_estim_error, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
        self.estimates = fit.params
        return fit
    
    def pairwise_ll(self, l, exog, params):
        '''Log likelihood function for every raw of data (sharing between countries).
        Return log likelihood.'''
        r, pw_nr = exog[0], exog[1]  # Distance between populations
        l = np.array(l)  # Make l an Numpy vector for better handling
        
        #print("Distance:")
        #print(r)
        #print("Blocks")
        #print(l)
        #print("Pairwise Number")
        #print(pw_nr)
        
        bins = self.mid_bins[self.min_ind:self.max_ind + 1] - 0.5 * self.bin_width  # Rel. bin edges
        l = l[(l >= bins[0]) * (l <= bins[-1])]  # Cut out only blocks of interest
        
        self.calculate_thr_shr(r, params)  # Calculate theoretical sharing PER PAIR
        self.calculate_full_bin_prob()  # Calculate total sharing PER PAIR /TM Matrix and FP-rate dont need update
        shr_pr = self.full_shr_pr[self.min_ind:self.max_ind]
        
        #print("Extracted Shr. Pr.:")
        #print(shr_pr)
        
        log_pr_no_shr = -np.sum(shr_pr) * pw_nr  # The negative sum of all total sharing probabilities
        if len(l) > 0:
            indices = np.array([(bisect_left(bins, x) - 1) for x in l])  # Get indices of all shared blocks
            l1 = np.sum(np.log(shr_pr[indices]))
        else: l1 = 0
        ll = l1 + log_pr_no_shr
        #print("LL:")

        return(ll)    
    
    def create_bins(self):
        '''Creates the bins according to parameters'''
        bins = np.arange(self.min_b, self.max_b, self.bin_width)  # Create the actual bins
        
        self.min_ind = bisect_left(bins, self.min_len)  # Find the indices of the relevant points
        self.max_ind = bisect_left(bins, self.max_len)
        self.mid_bins = bins + 0.5 * self.bin_width
        k = len(self.mid_bins)
        self.trans_mat = np.zeros((k, k)).astype(float)  # Create empty transition matrix
        
    def calculate_thr_shr(self, r, params):
        '''Calculates the expected Bessel-Decay per bin''' 
        # bd = bessel_decay_dens(self.mid_bins, r, C, sigma, mu)
        bd = self.block_shr_density(self.mid_bins, r, params)
        self.theoretical_shr = bd * self.bin_width  # Normalize for bin width (in cm)
        
    def calculate_trans_mat(self):
        '''Calculate the transition matrix from true estimated to
        observed values for block sharing.'''
        k = len(self.trans_mat)
        for i in range(k):  # Iterate over all starting values
            x = self.mid_bins[i]
            pr_detect = (1 - censor_prob((x)))  # Probability of detecting block
            for j in range(0, i):
                y = self.mid_bins[j]
                # Down probability conditional on bigger than cut off:
                trans_pr = prob_down(x) * down_rate(x) * np.exp(-down_rate(x) * (x - y)) / (1 - np.exp(-down_rate(x) * (x - 1)))
                self.trans_mat[j, i] = pr_detect * trans_pr * self.bin_width 
                              
            for j in range(i + 1, k):
                y = self.mid_bins[j]
                trans_pr = (1 - prob_down(x)) * up_rate(x) * np.exp(-up_rate(x) * (y - max(x, 1)))     
                self.trans_mat[j, i] = pr_detect * trans_pr * self.bin_width
            
            y = x  # Now do the i,i case
            trans_pr_d = prob_down(x) * down_rate(x) * np.exp(-down_rate(x) * (x - y)) / (1 - np.exp(-down_rate(x) * (x - 1)))
            # trans_pr_u = (1 - prob_down(x)) * up_rate(x) * np.exp(-up_rate(x) * (y - max(x, 1)))
            trans_pr_u = (1 - prob_down(x)) * up_rate(x) * np.exp(-up_rate(x) * (y - x))
            self.trans_mat[i, i] = pr_detect * 1 / 2.0 * (trans_pr_d + trans_pr_u) * self.bin_width  # Prob of not going anywhere         
        
    def calculate_full_bin_prob(self):
        '''Calculate the full probablities per bin'''
        # Transition matrix times theoretically expected + false positives.
        if self.error_model == True:
            self.full_shr_pr = (np.dot(self.trans_mat, self.theoretical_shr) + self.fp_rate)  # Model with full error
        else:
            self.full_shr_pr = self.theoretical_shr  # Model without any error in detection
    
    def get_bl_shr_interval(self, interval, r, params=[0, ]):
        '''Return the estimated block-sharing under the model in interval given distance r.
        For this use all the bins intersecting the interval and average.
        Assumes r is array and return array'''
        if params[0] == 0:  # If not parameters given use last ones fit
            params = self.estimates
        # Find the indices of the right ultimate bins:
        bins = self.mid_bins - 0.5 * self.bin_width
        ind = bisect_right(bins, interval[0])
        ind1 = bisect_left(bins, interval[1])
        
        estims = np.array([0.0 for i in r])
        for i in range(len(r)):
            self.calculate_thr_shr(r[i], params)  # Calculate the theoretical sharing
            self.calculate_full_bin_prob()  # Calculate the bin probability of sharing a block                
            # Do the numerical "Integral":
            mean_value = np.mean([self.full_shr_pr[j] for j in range(ind - 1, ind1 + 1)])
            estims[i] = mean_value * (interval[1] - interval[0]) / self.bin_width  # Normalize
        return(estims) 
            
    def block_shr_density(self, l, r, params):
        '''Returns block sharing density per cM; if l vector return vector
        Uses self.density_fun as function'''
        return self.density_fun(l, r, params)

############# Functions the class uses for calculating errors. From Ralph/Coop 2013.      

def censor_prob(l):
    '''Probability of being unobserved given true length of x'''
    return 1.0 / (1 + 0.0772355 * (l ** 2) * np.exp(0.5423082 * l))
        
def prob_down(l):
    '''Probability  the observed block is shorter than the true block'''
    l1 = max((l - 1, 0))
    return (1 - 1 / (1.0 + 0.5066205 * l1 * np.exp(0.6761991 * l1))) * 0.341945
    
def up_rate(l):
    '''parameter for (conditioned) exponential distr'n of observed-true 
    length given true length of x if observed > true'''
    return 1.399283
    
def down_rate(l):
    '''parameter for (conditioned) exponential distr'n of observed-true 
    length given true length of x if observed < true
    '''
    return np.min([12.0, (0.4009342 + 1.0 / (0.18161222 * l))])

def fp_rate(l):
    '''Gives the false positive rate per pair (!). If l vector return vector'''
    return np.exp(-13.704 - 2.095 * l + 4.381 * np.sqrt(l)) * 3587  # 3587 Centimorgans

def bessel_decay_dens(l, r, C, sigma, mu=0):
    '''Gives Bessel-Decay density per cM (!) If l vector return vector'''
    # l = 2.0 / (1.0 / interval[0] + 1.0 / interval[1])  # Calculate Harmonic Mean of interval
    l_e = l - mu / 2.0  # Update for population growth!
    l_e = l_e.clip(0.0)  # Update for very short block sizes 
    # l_e = max([l_e, 0]) 
    b_l = C * r ** 2 / (2 * l_e / 100.0 * sigma ** 2) * kv(2, np.sqrt(2 * l_e / 100.0) * r / sigma)
    return b_l / 100.0  # Factor in density for centi Morgan!

def dd_density(l, r, params):
    '''Gives the Doomsday density per cM(!) If l vector return vector'''
    C = params[0]
    sigma = params[1]
    b_l = C * r ** 3 / (4.0 * np.sqrt(2) * (l / 100.0 * sigma ** 2) ** (3 / 2.0)) * kv(3, np.sqrt(2.0 * l / 100.0) * r / sigma)
    return b_l / 100.0  # Factor for density in centi Morgan

def uniform_density(l, r, params):
    '''Gives density per cM(!) for constant population size. If l vector return vector'''
    C = params[0]
    sigma = params[1]
    b_l = C * r ** 2 / (2.0 * (l / 100.0 * sigma ** 2)) * kv(2, np.sqrt(2.0 * l / 100.0) * r / sigma)
    return b_l / 100.0  # Factor for density in centi Morgan  



####################################################################################################################################
class MLE_Estim_Barrier(MLE_estim_error):
    '''
    Class for MLE estimation of block sharing
    between populations with modeled error
    Bins into length blocks and models block-sharing
    between populations as indep. Poisson
    '''
    # Bins for mle_multi_run. Everything in cM!!
    min_b, max_b = 0, 30.1  # Minimum/Maximum bin for mle_multi_run    # 30.2
    bin_width = 0.1  # Bin width for mle_multi_run
    min_len, max_len = 4.0, 20.0  # Minimum/maximum bin length actually analyzed    #20
    min_ind, max_ind = 0, 0  # Indices for start stop of bins of interest
    mid_bins = []  # Array for the bins
    
    g = 1.0  # Chromosome Length
    fp_rate = []  # Array for false positives
    theoretical_shr = []  # Array for theoretical expected sharing per bin
    trans_mat = np.zeros((2, 2))  # Transition matrix for theor. to expected block-sharing
    full_shr_pr = []  # Array for the full bin sharing 
    
    th_shr = []  # Where the Vector for th_sharing will go (sharing per mid bin * bin width)
    density_fun = 0  # function used to calculate the block sharing density; is required to be per cM!!
    start_params = []  # List of parameters for the starting array
    error_model = True  # Parameter whether to use error model
    estimates = []  # The last parameter which has been fit
    diploid_factor = 4
    
    # Parameters for numerical calculation:
    barrier_pos = [0,0] # The Translation of the Barrier Center Point!
    barrier_angle = 0  # In Radiant!
    coords_bary = []
    step = 0
    L = 0
    
    
    # Maybe inherit from full likelihood object; as it is so different...
    def __init__(self, position_list, start_params, pw_IBD, pw_nr,
                 error_model=True, g=35.374, diploid=True, coarse=0.1, 
                 prior_sigma=0, barrier_pos=[0,0], barrier_angle=0, **kwds):
        '''Take position list and start parameters as input. 
        List of pw. nr and list of pw. IBD-Lists (in cM)
        g: Chromosome Length (in centiMorgan)'''
        exog = pw_nr  # Here endog is pairwise Nr
        endog = pw_IBD  # Endog 
        self.initialize_ll_model(endog, exog, **kwds)  # Initializes Likelihood Model
        self.create_bins()  # Create the Mid Bin vector and self.min_ind and self.max_ind
        self.fp_rate = fp_rate(self.mid_bins) * self.bin_width  # Calculate the false positives per bin (in cM).
        self.start_params = start_params 
        self.error_model = error_model  # Whether to use error model
        self.g = g
        self.diploid_factor = 4  # For Block Sharing in Diploids
        if diploid == False:
            self.diploid_factor = 1  # Block Sharing in Haploids
        if self.error_model == True:  # In case required:  
            self.calculate_trans_mat()  # Calculate the Transformation matrix
            
        if prior_sigma==0:  # If no Prior Sigma given; overwrite it with starting values:
            prior_sigma= start_params[1]
            
        self.barrier_pos=barrier_pos
        self.barrier_angle=barrier_angle
        # Calculates Raphael's stuff:
        position_list = centering_positions(position_list, [self.barrier_pos, self.barrier_angle])
        print("Barrier Pos.:")
        print(self.barrier_pos)
        print("Barrier Angle:")
        print(self.barrier_angle)
        print("Transformed Position List:")
        print(position_list)
        self.coords_bary, self.step, self.L = prepare_step_size(position_list, prior_sigma, coarse=coarse)
        # print self.step, self.L
        # print '-----------------------------'
        print("Barycentric Coordinates:")
        print(self.coords_bary)
        
        
        
    def loglikeobs(self, params):
        '''Calculate LL vector for every observation.
        Here: 1) Precalculate full sharing Matrix lxnxn
        2) Send it to pairwise_ll
        Return this vector'''
        pw_nr, pw_IBD = self.exog, self.endog # Pw Number is array of arrays.... WTF.
        for i in range(len(params)):
            print("Parameter %.0f : %.8f" % (i, params[i]))
        # print("Length of pw. IBD-sharing Vec: %i" % len(pw_IBD))
        # print("Length of pw. Number Vec: %i " % len(pw_nr))
            
        # Fit constant population density for testing:
        # print params
        
        # Uncomment this if you want to find out about all parameters:
        n = params[0:2]  # Density Parameter
        sigma = params[2:4]  # Dispersal Parameter
        #n =  np.array([params[0], params[0]])
        #sigma= np.array([params[1], params[1]])
        
        beta = 0 # params[5] # growth exponent. Change this to estimate it as well
        
        if np.min([n, sigma]) < 0:  # If Parameters do not make sense return infinitely negative likelihood
            return -np.ones(len(self.endog)) * (np.inf)
        
        # Calculate Full Block-Sharing Probability Matrix.
        # print("Calculating Sharing Matrix.")
        tic = time()
        mid_bins = self.mid_bins / 100.0  # Switch to MORGAN!!!
        
        th_mat = self.diploid_factor * self.g * ibd_sharing(self.coords_bary, self.L, self.step, mid_bins, sigma=sigma,
                                        population_sizes=n, pw_growth_rate=beta, max_generation=200)  # Factor 4 is for Diploids!!
        
        #print("Mid Bins")
        #print(mid_bins)
        
        #print("Genome Length:")
        #print(self.g)
        
        #print("Diploid Factor: ")
        #print(self.diploid_factor)
        
        #print("Error Model")
        #print(self.error_model)
        
        
        # Important: Factor for centimorgan!!!!!!
        self.th_shr = th_mat * self.bin_width / 100.0  # Normalize for bin width (in cM; as bins are in centiMorgan!!!)
        
        toc = time()
        print("Time for calculation: %.4f" % (toc - tic))
        
        # Which indices in sharing matrix to access:
        n = len(th_mat[0, 0, :])  # Nur of Countries where one compares to.
        assert(n * (n - 1) / 2 == len(pw_nr))
        assert(len(self.th_shr[:, 0, 0]) == len(self.mid_bins))
        
        xi, yi = np.tril_indices(n, -1)  # Gives the corresponding indices to array in matrix of thr. sharing
        
        # Depict pairwise Distances:
        # for i in xrange(len(xi)):
        #    pos1 = self.positions[xi[i]]
        #    pos2 = self.positions[yi[i]]
        #    pw_dist=np.sqrt(np.sum((pos1-pos2)**2))
        #    print("PW-Dist: %.4f" % pw_dist)
            
        
        # Calculate the Full lxnxn Sharing Matrix based on Raphael's Formula
        #### Do some work here. Work in progress!!
        # print("Position C1:")
        # print(self.positions[xi[0]])
        # print("Position C2:")
        # print(self.positions[yi[0]])
        ll = [self.pairwise_ll(pw_IBD[i], pw_nr[i][0], self.th_shr[:, xi[i], yi[i]]) for i in range(len(self.endog))]
        
        print("Total log likelihood: %.4f" % np.sum(ll))
        return np.array(ll).astype('float')  # Return negative log likelihood
                
    def pairwise_ll(self, l, pw_nr, thr_shr_pr):
        '''Log likelihood function for every raw of data (sharing between countries).
        Return log likelihood.
        l: Block List in CentiMorgan.
        pw_nr: Integer Number of Nr of pairwise comparisons
        Input: Vector of shared blocks; Block Sharing Probabilities'''
        l = np.array(l)  # Make l an Numpy vector for better handling
        
        # First Calculate full sharing probabilities per bin:
        if self.error_model == True:
            self.full_shr_pr = (np.dot(self.trans_mat, thr_shr_pr) + self.fp_rate)  # Model with full error.
        else:
            self.full_shr_pr = thr_shr_pr  # Model without any error in detection
        
        # Some tests which are checked:
        #print("Blocks:")
        #print(l)
        #print("Full sharing Probability:")
        #print(thr_shr_pr)
        
        # Extract all relevant Parameters.
        bins = self.mid_bins[self.min_ind:self.max_ind + 1] - 0.5 * self.bin_width  # Rel. bin edges
        l = l[(l >= bins[0]) * (l <= bins[-1])]  # Cuts out only blocks of interest
        
        
        shr_pr = self.full_shr_pr[self.min_ind:self.max_ind]
        log_pr_no_shr = -np.sum(shr_pr) * pw_nr  # The negative sum of all total sharing probabilities
        if len(l) > 0:
            indices = np.array([(bisect_left(bins, x) - 1) for x in l])  # Get indices of all shared blocks
            l1 = np.sum(np.log(shr_pr[indices]))
        else: l1 = 0
        ll = l1 + log_pr_no_shr
        
        # Pause to see everything (For Debugging!)
        #print("Extracted Shr. Pr.")
        #print(shr_pr)
        #print(ll)
        #raw_input()
        return(ll)   
    



####################################################################################################################################

   
######################### Some lines to test the code and make some plots
if __name__ == "__main__":
    test = MLE_estim_error(dd_density, [0, 0])
    test.calculate_thr_shr(120, [0.0024, 60.0])
    test.calculate_full_bin_prob()
    
    plt.figure()
    plt.plot(test.mid_bins, test.theoretical_shr, 'bo', label="Theoretical")
    plt.plot(test.mid_bins, test.fp_rate, 'ro', label="False positive rate") 
    plt.yscale('log')
    plt.legend()
    plt.show()  
            
    plt.plot(test.mid_bins, test.theoretical_shr, 'ro', label="Theoretical")
    plt.plot(test.mid_bins, test.full_shr_pr, 'go', label="Full sharing")
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    print(np.sum(test.trans_mat[:, 100]))  # Check whether Transition matrix is okay.
    # plt.plot(test.mid_bins, test.trans_mat[:, 100], 'ro')
    for i in range(1, 103, 10):
        plt.plot(test.mid_bins, test.trans_mat[:, i], label=str(test.mid_bins[i]) + "cM")
        plt.xlim([0, 16])
        # plt.plot(test.mid_bins, test.trans_mat[:,i],'ko')
    plt.xlabel("True Sharing", fontsize=20)
    plt.ylabel("Observed Sharing", fontsize=20)
    plt.legend()
    plt.show()
    
##############################
