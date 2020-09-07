import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import inv,eigh
from scipy import interpolate
import os

class GSFisher:

    def __init__(self,band = np.array([100,200]), nu_lims = np.array([110,200]),
    N_samples = 100, N_zbins = 20, int_time = 100*3600, N_poly=3, 
    fgnd_fid=np.array([np.log(320*1000),-2.54,-0.074,0.013])):
        '''
        Generates Fisher matrices for the 21cm signal, derivatives taken with 
        respect to the neutral fraction x_HI(z) for some redshift bins. Can 
        include foregrounds in two ways: parametrized and then marginalized over, 
        or included in the covariance. 

        Parameters:
        -----------

        band: 1d array
            The frequencies over which the experiment is conducted, [nu_min, nu_max]
            in MHz

        nu_lims: 1d array
            The frequencies that will be considered for the derivatives in MHz. 
            
            Not necessarily the same as the band; sometimes the experiment will 
            be conducted over a frequency band (which will afftect the foreground
            parametrization, for instance), but we'll only consider some frequencies
            in that band for the fisher matrix computation.

        N_samples: int
            How many temperature samples are measured in the frequency range set 
            by nu_lims

        N_zbins: int
            Number of redshift bins

        int_time: float/int
            Integration time of experiment in seconds 

        N_poly: int 
            Polynomial degree of foreground fit 

        fgnd_fid: 1d array 
            The fiducial parameters for the foreground polynomial fit, in order 
            [a_0, a_1, ...]. Default from Pritchard and Loeb 2010, arxiv 1005.4057
            for a bandwidth of [100,200] MHz


        Methods:
        --------

        compute_fisher_fgnds_marginalized:
            Method for including foregrounds as parameters, then marginalizing over 

        compute_fisher_fgnds_cov: 
            Method for including foregrounds in covariance. 
        '''

        #initializing attributes

        self.NU_21CM = 1420 #MHz
        self.band = band
        self.N_samples = N_samples
        self.N_zbins = N_zbins
        self.int_time = int_time
        self.nu_lims = nu_lims

        self.N_poly = N_poly 
        self.fgnd_fid = fgnd_fid 
        
        #---------------- frequency linspace and attributes ----------------# 
        self.nu_0 = np.mean(self.band)
        self.nu_linspace = np.flip(np.linspace(self.nu_lims[0], self.nu_lims[1],self.N_samples))
        self.bandwidth = self.band[1] - self.band[0]
        
        #---------------- redshift linspace and attributes ----------------#
        self.z_limits = (self.NU_21CM - self.nu_lims)/self.nu_lims 
        self.z_linspace = np.linspace(self.z_limits[1], self.z_limits[0], self.N_samples)
        self.Delta_z = self.z_limits[0]-self.z_limits[1]
        
    #====================== METHODS THAT RETURN FUNCTIONS =====================#
        
    def compute_fisher_fgnds_marginalized(self, const_xHI = False, const_xHI_val = 0.5,
    linear = False, zend = 5.5, z_start = 18, spline = False, spline_fct = None,
    delta_z = 1, z_mid =8):
        '''
        Method for computing Fisher matrix with marginalized foreground

        Parameters:
        -----------

        *ionization history flags:
            const_xHI: constant ionization history
            const_xHI_val: value of constant ionization history
            linear: linear ionization history
            zend,zstart: redshifts for which xHI=1 and =0 resp.
            spline: cubic spline ionization history
            spline_fct: function to evaluate spline, xHI(z)
            delta_z, z_mid: parameters for tanh ionization



        Returns:
        --------
        fisher matrix marginalized over foreground parameters
        '''
        
        #---- attributes determining which ionization model is considered ----#
        self.const_xHI = const_xHI
        self.const_xHI_val = const_xHI_val

        self.linear = linear
        self.zend = zend 
        self.z_start = z_start

        self.spline = spline 
        self.spline_fct = spline_fct

        self.delta_z = delta_z 
        self.z_mid = z_mid
        #---------------------------------------------------------------------#
        self.fgnds_in_cov = False #foreground flag
        self.get_fisher()

        marginalized = inv(inv(self.fisher)[:-4,:-4]) #marginalizing
        
        return marginalized


    def compute_fisher_fgnd_cov(self, const_xHI = False, const_xHI_val = 0.5,
    linear = False, zend = 5.5, z_start = 18, spline = False, spline_fct = None,
    delta_z = 1, z_mid =8, assert_Cfg_posdef=False):
        '''Method for computing fisher matrix with foregrounds included in covariance

        Parameters:
        -----------

        *ionization history flags:
            const_xHI: constant ionization history
            const_xHI_val: value of constant ionization history
            linear: linear ionization history
            zend,zstart: redshifts for which xHI=1 and =0 resp.
            spline: cubic spline ionization history
            spline_fct: function to evaluate spline, xHI(z)
            delta_z, z_mid: parameters for tanh ionization

        assert_Cfg_posdef: Bool
            if True, the foreground covariance will be reconstructed according to
            C = \sum_i \lambda_i < u_i^T u_i > where {lambda_i, u_i} are 
            eigenpairs of the foreground covariance, and the lambda_i < 0 are 
            excluded to ensure C_fg is positive definite. 

        Returns:
        --------
        fisher matrix with foregrounds included in covariance

        '''
        #---- attributes determining which ionization model is considered ----#
        self.const_xHI = const_xHI
        self.const_xHI_val = const_xHI_val

        self.linear = linear
        self.zend = zend 
        self.z_start = z_start

        self.spline = spline 
        self.spline_fct = spline_fct

        self.delta_z = delta_z 
        self.z_mid = z_mid
        #------------------------ foreground paramters ------------------------#
        self.nu_star = 150 # MHz
        
        #first entry corresponds to synchrotron, second to free-free emission
        self.A = [335.4*1000, 33.5*1000]#mK 
        self.alpha= [2.8,2.15] 
        self.Delta_alpha = [0.1,0.01] 

        #----------------------------------------------------------------------#
        
        self.fgnds_in_cov = True
        self.assert_Cfg_posdef= assert_Cfg_posdef

        self.get_fisher()

        return self.fisher

    #======================= fisher matrix computation ========================#
    def get_fisher(self):

        self.get_dT_b() #getting deriv of brigthness temp wrt x_HI
        self.get_redshift_bins() #getting bin edges
        
        if self.fgnds_in_cov: #getting dimensions of fisher matrix
            self.fisher_dims = self.N_zbins
        else:
            self.fisher_dims = self.N_zbins + self.N_poly + 1

        self.fisher = np.zeros((self.fisher_dims,self.fisher_dims))
        self.get_cov() #getting covariance matrix

        for alpha in range(self.fisher_dims):
            for beta in range(self.fisher_dims):
                #getting F_{alpha,beta}, rounding to ensure symmetry
                self.fisher[alpha, beta] = np.round(self.get_entry(alpha, beta),6)

    def get_entry(self,a,b):
        '''gets array of temperatures to be summed over, and sums them up (with
        pretty linear algebra magic) to get the matrix entry'''

        a_arr = self.get_array(a)
        b_arr = self.get_array(b)
        
        entry = np.dot(a_arr,np.dot(self.cov_inv,b_arr))

        return entry

    def get_array(self,param):
        '''getting the derivative arrays to be summed over for matrix entry'''
        arr = np.zeros(self.N_samples)
        
        if param < (self.N_zbins): #z bin parameter
            arr[self.inds_binned[param]] = self.dTb_binned[param]
            # only the derivatives dT_b(z_i) where z_i falls into the given 
            # redshift bin are nonzero 
        else:
            k = (self.N_poly+1)-(self.fisher_dims-param)
            arr = np.round(self.dT_fg(k))
        return arr

    #=========================== Binning functions ============================#    
    def get_redshift_bins(self):
        '''given the redshift range and number of bins, gets bin edges'''
    
        self.bin_edges = np.linspace(self.z_limits[1], self.z_limits[0], self.N_zbins+1)
        self.get_binned()

    def get_binned(self):
        '''given the array of redshifts and the redshift bins, determines which 
        indices of the array of derivatives fall into the bins. Generates:
        
        inds_binned: 2d array
            where inds_binned[0] is an array of the indices that fall into the 
            bin {bin_edges[0],bin_edges[1]} and etc

        dTb_binned: 2d array
            where dTb_binned[0] is an array of the derivatives of the brigtness
            temperature that fall into the bin {bin_edges[0],bin_edges[1]} and etc'''

        #this is UGLY. make nice one day
        bin_indices = [0]
        for bin_val in self.bin_edges[1:]:
            val = np.argmax( self.z_linspace > bin_val)
            if val == 0: #ie bin_val > r_max
                val = len(self.z_linspace)
            bin_indices.append(val)

        indices = np.arange(0,self.N_samples)
        indices_binned = []
        for i in range(len(bin_indices)-1):
            indices_binned.append(indices[bin_indices[i]:bin_indices[i+1]])

        self.inds_binned = np.array(indices_binned) 

        temps_binned = []
        for i in range(len(bin_indices)-1):
            temps_binned.append(self.dT_b[bin_indices[i]:bin_indices[i+1]])

        self.dTb_binned = np.array(temps_binned)
   
    #=================== Brightness temperature functions =====================#    

    def get_dT_b(self):
        '''analytic expression for the derivative of the brightness temp with 
        respect to x_HI '''

        self.dT_b = 27*np.sqrt((1+self.z_linspace)/10)


    def get_T_b(self):
        '''Gets brightness temperature for each sampled frequency'''
        T_21cm = 27 #mK
        
        if self.const_xHI:
            xHI = self.const_xHI_val*np.ones_like(self.z_linspace)
        
        elif self.linear: 
            slope = 1/(self.z_start - self.zend)
            xHI = self.z_linspace*slope - slope*self.zend

        elif self.spline:
            xHI = self.spline_fct(self.z_linspace)

        else: #tanh model
            xHI = 0.5*(np.tanh((self.z_linspace-self.z_mid)/self.delta_z) +1)

        
        return (T_21cm)*np.sqrt((1+self.z_linspace)/10)*xHI

    #======================= foreground model functions =======================#
    
    def dT_fg(self, k):
        '''Derivatives of foreground temperature, taking wrt a parameter in the 
        polynomial parametrization k'''

        poly_arg = np.log(self.nu_linspace/self.nu_0)
        poly = np.zeros(self.N_samples) 
        for i in range(self.N_poly+1):
            poly+= self.fgnd_fid[i]*(poly_arg**i)
            
        return self.fgnd_fid[k]*np.exp(poly)

    def get_T_fg(self):
        '''returns the foreground temperature with the polynomial model'''

        poly_arg = np.log(self.nu_linspace/self.nu_0)
        poly = np.zeros(self.N_samples)
        for i in range(self.N_poly+1):
            poly+= self.fgnd_fid[i]*(poly_arg**i)
            
        return np.exp(poly)

  
    #========================= Covariance functions ===========================#    

    def get_cov(self):
        '''gets the covariance matrix for the fisher matrix computation'''
       
        #-------------------------- noise covariance --------------------------# 
        self.T_sky = self.get_T_fg() + self.get_T_b()
        self.channel_bwidth = (self.nu_linspace[0]-self.nu_linspace[1])*1e6
        
        self.variance = (self.T_sky**2)/(self.channel_bwidth*self.int_time)
        self.C_noise = np.identity(self.N_samples)*self.variance

        #------------------------ foreground covariance -----------------------# 
        if self.fgnds_in_cov:
            self.get_fgnd_cov()
            covariance = self.C_noise + self.C_fg
        else:
            covariance = self.C_noise 

        self.cov = covariance
        self.cov_inv = inv(self.cov)

    def get_fgnd_cov(self):
        '''The foreground covariance is the sum of the covaraicnes for the 
        synchrotron emission foregrounds and free-free emission foregrounds'''

        self.C_synch = self.foreground_covariance(0)
        self.C_ff = self.foreground_covariance(1)
        
        C_fg = self.C_synch + self.C_ff

        if self.assert_Cfg_posdef:#taking care of positive-definiteness
            vals,vects = eigh(C_fg)
            vects = vects.T
            C = np.zeros_like(C_fg)
            for i,val in enumerate(vals):
                if val >0:
                    C += val*np.outer(vects[i],vects[i])
            self.C_fg = C 
        else:
            self.C_fg = C_fg
        
    def foreground_covariance(self, flag):
        '''Gets the covariance for the foreground emission.
        
        flag: int 
            determines which type of foreground. 
            0 --> synchrotron, 1 --> free-free
        '''

        cov = np.zeros((self.N_samples, self.N_samples))
        self.flag = flag
        
        for i in range(self.N_samples):
            for j in range(self.N_samples):
                cov[i,j] = np.round(self.C(self.nu_linspace[i], self.nu_linspace[j]))

        return cov

    def C(self,nu,nu_prime):
        '''
        expressions for the foreground covariance as described in Liu and Tegmark
        2012, MNRAS 419 3491-3504, equations 12-15
        '''
        arg = (nu*nu_prime)/(self.nu_star**2)
        exponent = -self.alpha[self.flag] + 0.5*(self.Delta_alpha[self.flag]**2)*np.log(arg)
        return (self.A[self.flag]**2)*(arg**exponent) - self.m(nu)*self.m(nu_prime)

    def m(self,v):
        arg = v/self.nu_star
        exponent = -self.alpha[self.flag] + 0.5*(self.Delta_alpha[self.flag]**2)*np.log(arg)
        return self.A[self.flag]*(arg**exponent)


class kSZ_Fisher:

    def __init__(self, ksz_variance_path, ksz_derivs, deriv_ells):
        '''
        Gets the Fisher matrix for the kSZ. 
        
        Parameters:
        -----------
        
        ksz_variance_path: str 
            Path to file containing digitized data for the variance of the kSZ
            as a function of ell 

        ksz_derivs: 2d array
            The derivatives of the kSZ taken wrt x_HI(z), ordered in increasing
            redshift. That is, if the redshift bins go from [z_min, z_max], then
            ksz_derivs[0] is the derivatives taken wrt x_HI(z_min).

        deriv_ells: 1d array
            The ell for which the derivatives are defined.  


        Methods:
        --------
        compute_ksz_fisher:
            Returns the Fisher matrix for the kSZ.
        '''

        #initializing attributes
        self.ksz_variance_path = ksz_variance_path
        self.ksz_derivs = ksz_derivs 
        self.deriv_ells = deriv_ells

    def compute_ksz_fisher(self):

        self.interpolate_ksz_variance()
        self.get_ksz_cov()
        self.get_fisher()

        return self.fisher

    def interpolate_ksz_variance(self):        
        '''Given digitized data, interpolates and creates a function var_func
        that will return the interpolated variance for any ell within the range 
        of the digitized data'''
        
        ksz_variance_data = np.loadtxt(self.ksz_variance_path).T

        self.ksz_variance_ells = ksz_variance_data[0]
        self.ksz_variance = ksz_variance_data[1] 
        self.var_func = interpolate.interp1d(self.ksz_variance_ells,self.ksz_variance)


    def get_ksz_cov(self):
        '''Gets the covaraince and inverse covariance from the interpolated data'''
        
        # the ksz derivs defined for more ell than the variance is, so we have 
        # adjust the range

        lower_cutoff_ind = 1
        upper_cutoff_ind = np.argwhere(self.deriv_ells < max(self.ksz_variance_ells))[-1][0]+1
        self.ells = self.deriv_ells[lower_cutoff_ind:upper_cutoff_ind]
        self.derivs = []
        for deriv in self.ksz_derivs:
            self.derivs.append(deriv[lower_cutoff_ind:upper_cutoff_ind])
        
        # the covariance is variance**2
        self.ksz_noise = self.var_func(self.ells)**2
        self.ksz_cov = np.eye(len(self.ells))*self.ksz_noise
        self.ksz_cov_inv = inv(self.ksz_cov)

    def get_fisher(self):
        '''computes the fisher matrix'''

        fisher_shape = len(self.ksz_derivs)
        fisher = np.zeros((fisher_shape, fisher_shape))

        for alpha in range(fisher_shape):
            for beta in range(fisher_shape):
                    fisher[alpha, beta] += self.get_fisher_entry(alpha,beta)

        self.fisher = (fisher + fisher.T)/2

    def get_fisher_entry(self,a,b):
        return np.dot(self.derivs[a],np.dot(self.ksz_cov_inv,self.derivs[b]))



def load_derivatives(path, const_xHI = False,perval = 0.1, log = True):
    '''Assumes directory structure: 
    
    results
    |
    └───results_0
    |   |   kSZ_power_spectrum_kSZ_Cells_zval_0_neg.txt
    |   |   kSZ_power_spectrum_kSZ_Cells_zval_0_pos.txt
    └───results_1
    |   |   kSZ_power_spectrum_kSZ_Cells_zval_1_neg.txt
    |   |   kSZ_power_spectrum_kSZ_Cells_zval_1_pos.txt
    | ...

    where kSZ_power_spectrum_kSZ_Cells_zval_i_neg.txt is the ksz power spectrum 
    evaluated with x_HI(z = zbin_i) -= delta-x_HI, and the postitive one is 
    evaluated with x_HI(z = zbin_i) += delta-x_HI. 

    This is ugly and too many things hardcoded. I will make it more general 
    later. 
    
    '''
    T_CMB_uK=2.7260*1e6
   
    specs_pos = []
    specs_neg = []
    zval_pos = []
    zval_neg = []

    for d in os.listdir(path):
        for f in os.listdir(path +"/"+d):
            loc = "/" + path +"/"+d+"/"+f
            zval = (int(f.split("_")[6]))
            per_type =(f.split("_")[-1].split(".")[0])
            
            l1, ksz_tot, ksz1 = np.loadtxt(loc,unpack=True)    
            patch = l1*(l1+1)*ksz1/(2*np.pi)*(T_CMB_uK**2)
            if per_type == "pos":
                specs_pos.append(patch)
                zval_pos.append(zval)
            elif per_type == "neg":
                specs_neg.append(patch)
                zval_neg.append(zval)
                
        ells = l1
                
    inds_pos = np.argsort(zval_pos)
    inds_neg = np.argsort(zval_neg)

    specs_pos_sorted = np.array(specs_pos)[inds_pos]
    specs_neg_sorted = np.array(specs_neg)[inds_neg]

    derivs = specs_pos_sorted - specs_neg_sorted

    #dividing by 2*delta-x_HI ; this part should not be hardcoded
    if log: 
        return ells, derivs
    else:
        if const_xHI:
            derivs /= 2*perval
        else:
            derivs[:39] /= 2*0.01
            derivs[39:49] /= 2*0.001
            derivs[49:] /= 2*0.0001 
        return ells,derivs