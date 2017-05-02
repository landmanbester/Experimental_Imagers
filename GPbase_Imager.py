# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:06:57 2016

@author: landman (Pirated from Benjamin Hugo's scripts)

GP imager with explicit basis functions

"""
import time
from numpy import load,reshape,savez,real,log,argwhere,diag,nan_to_num,tile,vstack,meshgrid,eye,linspace,array,sin,cos,empty,dot,deg2rad,arange,where,pi,ones,sinc,exp,zeros,fft,complex,max,sqrt,arcsin,arctan2,random,abs,log10,angle,copy
from scipy import optimize as opt
from scipy.signal import fftconvolve
from scipy.linalg import solve_triangular as soltri
from numpy.linalg import solve, slogdet, inv,eigh,cholesky,matrix_rank
from matplotlib import pyplot as plt

class python_imager(object):
    def __init__(self,npix,nsource,max_I, #Params for sky model
                 filter_half_support, filter_oversampling_factor, #Params for AA-filter
                 ref_ra,ref_dec,observation_length_in_hrs,integration_length, #Params for simulating uv coverage
                 filter_type='sinc',mode='Full'):
        #Set number of pixels
        self.Nx = npix
        self.Ny = npix

        #Initialise the instrumental setup
        self.setup()

        #Set up an anti-aliasing filter params
        self.set_AA_filter(filter_half_support, filter_oversampling_factor, filter_type)

        #Set sky and data (load_prior() loads a previously computed prior model)
        if mode=='Full':

            #Simulate a sky model
            self.sim_sky()

            #Simulate uv tracks of instrument
            self.sim_uv(ref_ra,ref_dec,observation_length_in_hrs,integration_length,plot_on=False)

            #Degrid to get model vis (should use DFT here)
            vis = self.fft_degrid(self.model_sky,mode="set")

            #Make the dirty image by gridding and iFFT
            self.grid_ifft(vis)

            #CLEAN the image with Hogbom CLEAN (Minor cycle only)
            self.ICLEAN = self.CLEAN_HOG(self.IDIRTY.copy(),self.psf.copy(),plot_on=True)

            #Fit the Gaussian process
            #First set basis function
            self.extract_point_source_loc()
            self.basis_h()
            self.set_prior(self.ICLEAN,train_theta=True)

            self.save_and_plot(SAVE=True,PLOT=True)

        elif mode=='Load':
            #Simulate uv tracks of instrument
            self.sim_uv(ref_ra,ref_dec,observation_length_in_hrs,integration_length,plot_on=False)

            #Load GP attributes
            self.load_prior()

        #Set beta (the parameter controlling acceptance rate)
        self.alpha = 0.001
        self.SIGMAslow = 0.01

    def abs_diff(self,x,xp):
        """
        Creates matrix of differences (x_i - x_j) for vectorising.
        """
        n = x.size
        np = xp.size
        return tile(x,(np,1)).T - tile(xp,(n,1))

    def lik_as_func_of_beta(self,beta):
        #Predict with given value of beta
        fmean = self.get_meanf(beta)
        #Compute the likelihood with the mean function
        vis = self.fft_degrid(fmean.reshape((self.Nx,self.Ny)),mode=1)
        return self.get_chi2(vis)        

    def get_chi2(self,vis):
        """
        Get likelihood of current sample
        """
        tmp = self.visdat - vis
        chi2 = sum(real(tmp*tmp.conj())/self.svisdatsq)
        return chi2

    def extract_point_source_loc(self):
        """
        This function should find the point sources in the CLEANED image using a source extractor.
        For now we simply include two sources at known locations (i,j) where i is
        the row pixel number and j the column pixel number (start counting at 0)
        """
        #One source at (30,45)
        self.k1 = 30*self.Ny + 45
        #And another at (50,55)
        self.k2 = 50*self.Ny + 55
        return

    def basis_h(self):
        """
        A basis of Dirac delta functions
        """
        b1 = zeros(self.Nx*self.Ny)
        b1[self.k1] = 1.0
        b2 = zeros(self.Nx*self.Ny)
        b2[self.k2] = 1.0
        self.H = vstack((b1,b2))
        return

    def get_meanf(self,beta):
        fmean = self.fmeanNP + dot(self.RP.T,beta)
        #Set any < 0 values to 0
        if (fmean<0.0).any():
            I = argwhere(fmean<0)
            fmean[I] = 0.0        
        return fmean

    def mean_and_covf(self,theta,y,XX,n):
        """
        Here we compute the posterrior mean, covariance and the values of the weights
        Note x = xp and XX = XXp = XXpp in this application
        """
        K = self.cov_func(theta,XX)
        L = cholesky(K)
        Linv = soltri(L.T,eye(n)).T
        Kp = self.cov_func(theta,XX,mode=1)
        HLinvT = dot(self.H,Linv.T)
        LinvKp = dot(Linv,Kp)
        R = self.H - dot(HLinvT,LinvKp)
        self.RP = R
        HKinvHTinv = inv(dot(HLinvT,HLinvT.T))
        beta = dot(HKinvHTinv,dot(HLinvT,dot(Linv,y)))
        self.fmeanNP = dot(LinvKp.T,dot(Linv,y))
        if (self.fmeanNP<0).any():
            print "There were fmeanNP < 0, setting them to zero"
            I = argwhere(self.fmeanNP<0)
            self.fmeanNP[I] = 0.0
        fmean = self.get_meanf(beta)
        fcov = Kp - dot(LinvKp.T,dot(Linv,y)) + dot(R.T,dot(HKinvHTinv,R)) #Note Kp = Kpp
        return beta,fmean,fcov

    def logp2(self,theta,XX,y,n):
        K = self.cov_func(theta,XX)
        L = cholesky(K)
        logdetK = 2.0*sum(log(diag(L)))
        Linv = soltri(L.T,eye(n)).T
        Linvy = solve(L,y)
        m = matrix_rank(self.H.T)
        HLinvT = dot(self.H,Linv.T)
        A = dot(HLinvT,HLinvT.T)
        logdetA = slogdet(A)[1]
        Ainv = inv(A)
        C = dot(Linv.T,dot(HLinvT.T,dot(Ainv,dot(HLinvT,Linv))))
        tmp = dot(Linvy.T,Linvy)/2.0 + logdetK/2.0 + (n-m)*log(2*pi)/2.0 + logdetA/2.0 -  dot(y.T,dot(C,y))/2.0
        print tmp
        return tmp

    def set_prior(self,yi,train_theta=True):
        """
        This function trains a GP to the CLEANED image. The GP serves as the prior.
        Input:  yi = CLEANED image
        """
        npix = yi.shape[0]
        #Set domian
        l = linspace(0,npix-1,npix)
        m = linspace(0,npix-1,npix)

        #Create grid and flatten
        X,Y = meshgrid(l,m)
        x = vstack((X.flatten(order='C'),Y.flatten(order='C')))

########################################remember we pass x**2 directly#######################################
        XX = x[0,:] + 1j*x[1,:]
        #XX = XXsq.get_xx(x,x,npix**2,npix**2,2)
        #XX = XXsq.get_xxsq(x,x,npix**2,npix**2,2) #Here we compute the square to reduce unnecessary operations
        XX = abs(tile(XX,(npix**2,1)).T - tile(XX,(npix**2,1)))**2

        #Flatten the image
        yif = yi.flatten(order='C')

        #Get optimal hypers
        if train_theta:
            #Set (somewhat reasonable) initial hyper
            sigmaf = 0.33467487
            ll = 3.9480936
            sigman = 0.92406035
            theta = array([sigmaf,ll,sigman])

            #Set bounds for optimiser
            bnds = ((1e-8, None), (1e-8, None), (1e-8,None))
            print "Fitting GP"
            thetap = opt.minimize(self.logp2,theta,args=(XX,yif,npix**2),method='L-BFGS-B',bounds=bnds)
            theta = thetap.x
            if thetap['success']:
                print "Optimised hypers = ", theta
                savez("theta.npz",theta = theta)
        else:
            thetaf = load("theta.npz")
            theta = thetaf["theta"]
            print 'sigmaf = ',theta[0], ' l = ', theta[1], ' sigman = ', theta[2]

        #Get mean, cov and weights
        self.beta,self.fmean,self.fcov = self.mean_and_covf(theta,yif,XX,npix**2)
            
        print "betas = ", self.beta
        #Now get eigendecomposition (for fast sampling)
        self.W,self.V = eigh(self.fcov)
        self.srtW = diag(nan_to_num(sqrt(nan_to_num(self.W))))
        return

    def load_prior(self):
        Prior = load("Sky_Prior.npz")
        self.fmean = Prior['fmean']
        self.fmeanNP = Prior['fmeanNP']
        self.RP = Prior['RP']
        self.beta = Prior['beta']
        self.fcov= Prior['fcov']
        self.W=Prior['W']
        self.srtW = diag(nan_to_num(sqrt(nan_to_num(self.W))))
        self.V=Prior['V']
        self.model_sky=Prior['model_sky']
        self.ICLEAN=Prior['ICLEAN']
        self.visdat=Prior['visdat']
        self.svisdatsq=Prior['svisdatsq']
        self.scaled_uv = Prior['scaled_uv']
        self.vis_uv = Prior['vis_uv']
        return

    def sample(self,f):
        """
        Returns pCN proposal for MCMC. For normal sample use simple_sample
        """
        f0 = f - self.fmean
        return self.fmean + sqrt(1-self.alpha**2)*f0 + self.alpha*self.V.dot(self.srtW.dot(random.randn(f0.size)))

    def simple_sample(self):
        """
        Returns sample from GPR posterior
        """
        return self.fmean + self.V.dot(self.srtW.dot(random.randn(self.Nx*self.Ny)))

    def cov_func(self,theta,x,mode="Noise"):
        """
        Covariance function including noise variance. Choose from the covariance functions below (make sure you uncomment the same covariance functions for both the K and K2 functions)
        We pass x^2 directly to avoid unnecessary sqrt computations
        """
        if mode != "Noise":
            #Squared exponential
            return theta[0]**2*exp(-x/(2*theta[1]**2))
            #Mattern 3/2
            #return theta[0]**2*exp(-sqrt(3)*abs(x)/theta[1])*(1 + sqrt(3)*abs(x)/theta[1]) + theta[2]**2*eye(x.shape[0])
            #Mattern 5/2
            #return theta[0]**2*exp(-sqrt(5)*abs(x)/theta[1])*(1 + sqrt(5)*abs(x)/theta[1] + 5*abs(x)**2/(3*theta[1]**2)) + theta[2]**2*eye(x.shape[0])
            #Mattern 7/2
            #return theta[0]**2*exp(-sqrt(7)*abs(x)/theta[1])*(1 + sqrt(7)*abs(x)/theta[1] + 14*abs(x)**2/(5*theta[1]**2) + 7*sqrt(7)*abs(x)**3/(15*theta[1]**3)) + theta[2]**2*eye(x.shape[0])
        else:
            #Squared exponential
            return theta[0]**2*exp(-x/(2*theta[1]**2)) + theta[2]**2*eye(x.shape[0])
            #Mattern 3/2
            #return theta[0]**2*exp(-sqrt(3)*abs(x)/theta[1])*(1 + sqrt(3)*abs(x)/theta[1]) + theta[2]**2*eye(x.shape[0])
            #Mattern 5/2
            #return theta[0]**2*exp(-sqrt(5)*abs(x)/theta[1])*(1 + sqrt(5)*abs(x)/theta[1] + 5*abs(x)**2/(3*theta[1]**2))
            #Mattern 7/2
            #return theta[0]**2*exp(-sqrt(7)*abs(x)/theta[1])*(1 + sqrt(7)*abs(x)/theta[1] + 14*abs(x)**2/(5*theta[1]**2) + 7*sqrt(7)*abs(x)**3/(15*theta[1]**3)) + theta[2]**2*eye(x.shape[0])

    def dcov_func(self,theta,x,mode=0):
        """
        We pass x^2 directly to avoid unnecessary sqrt computations
        """
        if mode == 0:
            return 2*theta[0]*exp(-x/(2*theta[1]**2))
        elif mode == 1:
            return x*theta[0]**2*exp(-x/(2*theta[1]**2))/theta[1]**3
        elif mode == 2:
            return 2*theta[2]*eye(x.shape[0])

    def twoD_Gaussian(self,(x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)
        a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
        b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
        c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.flatten()

    def fit_2D_Gaussian(self,PSF,plot_on=True):
        """
        Fit an elliptical Gaussian to the primary lobe of the PSF
        """
        #Get the full width at half maximum height of the PSF
        I = argwhere(PSF>=0.5*PSF.max())

        #Create an array with these values at the same indices and zeros otherwise
        lk,mk = PSF.shape
        psf_fit = zeros([lk,mk])
        psf_fit[I[:,0],I[:,1]] = PSF[I[:,0],I[:,1]]

        # Create x and y indices
        x = linspace(0, PSF.shape[0]-1, PSF.shape[0])
        y = linspace(0, PSF.shape[1]-1, PSF.shape[1])
        x, y = meshgrid(x, y)

        # add some noise to the data and try to fit the data generated beforehand
        initial_guess = (0.5,lk/2,mk/2,1.75,1.4,-4.0,0)
        #Flatten the data
        data = psf_fit.ravel()
        #Fit the function (Gaussian for now)
        popt, pcov = opt.curve_fit(self.twoD_Gaussian, (x, y), data, p0=initial_guess)
        #Get function with fitted params
        data_fitted = self.twoD_Gaussian((x, y), *popt)
        #Normalise the psf to have a max value of one
        data_fitted = data_fitted/data_fitted.max()
        #Plot the data
        if plot_on:
            plt.figure(figsize=(25, 25))
            plt.subplot(131)
            plt.title("PSF")
            plt.imshow(PSF, cmap="gray", origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
            plt.xlabel(r"$l$",fontsize=18)
            plt.ylabel(r"$m$",fontsize=18)
            plt.subplot(132)
            plt.title("PSF_trunc")
            plt.imshow(psf_fit, cmap="gray", origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
            plt.xlabel(r"$l$",fontsize=18)
            plt.ylabel(r"$m$",fontsize=18)
            plt.subplot(133)
            plt.title("PSF_fit")
            plt.imshow(data_fitted.reshape(lk,mk), cmap="gray", origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
            plt.xlabel(r"$l$",fontsize=18)
            plt.ylabel(r"$m$",fontsize=18)
            plt.tight_layout()
        return data_fitted.reshape(lk,mk)

    def CLEAN_HOG(self,ID,PSF,gamma = 0.1,threshold = "Default", niter = "Default", plot_on=True):
        """
        This is a simple Hogbom CLEAN assuming a full cleaning window
        Input:  ID = the dirty image to be cleaned
                PSF = the point spread function
                gamma = the gain factor (must be less than one)
                theshold = the threshold to clean up to
                niter = the maximum number of iterations allowed
        Output: ICLEAN = the cleaned image
                ID = the residuals
        """
        #Check that PSF is twice the size of ID
        if PSF.shape[0] != 2*ID.shape[0] or PSF.shape[1] != 2*ID.shape[1]:
            print "Warning PSF not right size"

        #Initialise array to store cleaned image
        npix = ID.shape[0] #Assuming square image
        ICLEAN = zeros([npix,npix])

        if niter == "Default":
            niter = 5*npix

        #Get indices where ID is max
        p,q = argwhere(ID==ID.max()).squeeze()
        pmin,qmin = argwhere(ID==ID.min()).squeeze()
        #Initialise Istar and counter
        Istar = ID[p,q]

        if threshold=="Default":
            threshold = 0.1*Istar #Imin + 0.001*(Istar - Imin)
            print "Threshold set at ", threshold
        else:
            print "Assuming user set threshold"

        #CLEAN the image
        i = 0 #counter index
        while Istar > threshold and i <= niter:
            #First we set the
            ICLEAN[p,q] += Istar*gamma
            #Subtract out pixel
            ID -= gamma*Istar*PSF[npix - p:2*npix - p,npix - q:2*npix - q]
            #Get new indices where ID is max
            p,q = argwhere(ID==ID.max()).squeeze()
            #Get max value
            Istar = ID[p,q]
            #Increment counter
            i += 1
        #Warn if niter exceeded
        if i > niter:
            print "Warning: number of iterations exceeded"
            print "Minimum ID = ", ID.max()
        else:
            print "Converged in %s iterations" % (i)

        #get the ideal beam (fit 2D Gaussian to HWFH of PSF)
        psf_ideal = self.fit_2D_Gaussian(PSF,plot_on=False)

        #Now convolve ICLEAN with ideal beam
        #ICLEAN2 = convolve(ICLEAN,psf_ideal,mode='constant', cval=0.0)  #Slow direct
        self.IPREDICT = fftconvolve(ICLEAN,psf_ideal,mode='same') #, cval=0.0) #Fast using fft

        #Finally we add the residuals back to the image
        self.residual = ID
        #ICLEAN2 += ID
        return ICLEAN

#    def get_DFT(self,Image,mode='Set'):
#        #First compute the kernels since they can be reused each time
#        if mode=='Set':
#            #set u
#            u = reshape(self.uvw[0],(self.nvis,1))
#            #set v
#            v = reshape(self.uvw[1],(self.nvis,1))
#            #set w
#            w = reshape(self.uvw[2],(self.nvis,1))
#            #Compute the kernel
#            K = np.exp(-1j*2.0*np.pi*(l*u + m*v))
#            #return K.dot(x)
#        for i in range(self.nvis):
#            self.vis_tmp[i] = sum(self.uvw[i,0]*l[i])
#
#
#        return

    def sim_sky(self):
        """
        This function simulates the model sky (for now only point sources)
        Inputs: npix = number of pixels for the image
                nsource = number of point sources
                max_I = maximum intensity of points sources

        Attributes: model_sky = the model image
                    model_vis = model visibilities
                    Nx = number of pixels in x
                    Ny = number of pixels in y
        """
        #Set blank image
        self.model_sky= zeros([npix,npix])

        #First throw down some sort of extended emmision
        l = arange(0,self.Nx,1)
        m = arange(0,self.Ny,1)
        ll,mm = meshgrid(l,m)
        self.model_sky[:,:] = self.twoD_Gaussian((ll, mm), 10.0, 15, 25, 2.0, 4.5, 60, 0.0).reshape((npix,npix))

        #Now simulate some point sources (some of which may overlap extended emission)
        #One point source at (30,45)
        self.model_sky[30,45] = 25.0
        #And one at (50,55)
        self.model_sky[50,55] = 21.0
        return

    def sim_uv(self,ref_ra, ref_dec,observation_length_in_hrs,integration_length,plot_on=True):
        """
        Simulates uv coverage given antenna coordintes in the East-North-Up frame

        Keyword arguments:
        ref_ra --- Right Ascension of pointing centre (degrees)
        ref_dec --- Declination of pointing centre (degrees)
        integration_length --- Integration length in hours
        plot_on --- Plots the projected u,v coverage after simulation (default=True)

        Attributes: uvw = the (u,v,w) coordinates of the instrument
        """
        no_antenna = self.ENU.shape[0]
        no_baselines = no_antenna * (no_antenna - 1) // 2 + no_antenna
        cphi = cos(deg2rad(self.ARRAY_LATITUDE))
        sphi = sin(deg2rad(self.ARRAY_LATITUDE))
        reference_ra_rad = deg2rad(ref_ra)
        reference_dec_rad = deg2rad(ref_dec)
        integration_length_in_deg = integration_length / 24.0 * 360.0
        no_timestamps = int(observation_length_in_hrs / integration_length)
        row_count = no_timestamps * no_baselines
        self.nvis = row_count

        l = no_antenna
        k = no_antenna
        self.uvw = empty([row_count,3])

        for r in range(0,row_count):
            timestamp = r / (no_baselines)
            baseline_index = r % (no_baselines)
            increment_antenna_1_coord = (baseline_index / k)

            # calculate antenna 1 and antenna 2 ids based on baseline index using some fancy
            # footwork ;). This indexing scheme will enumerate all unique baselines per
            # timestamp.

            l -= (1) * increment_antenna_1_coord
            k += (l) * increment_antenna_1_coord
            antenna_1 = no_antenna-l
            antenna_2 = no_antenna + (baseline_index-k)
            new_timestamp = ((baseline_index+1) / no_baselines)
            k -= (no_baselines-no_antenna) * new_timestamp
            l += (no_antenna-1) * new_timestamp
            #conversion to local altitude elevation angles:
            be,bn,bu = self.ENU[antenna_1] - self.ENU[antenna_2]
            mag_b = sqrt(be**2 + bn**2 + bu**2)
            epsilon = 1e-12
            A = arctan2(be,(bn + epsilon))
            E = arcsin(bu/(mag_b + epsilon))
            #conversion to equitorial coordinates:
            sA = sin(A)
            cA = cos(A)
            sE = sin(E)
            cE = cos(E)
            Lx = (cphi*sE-sphi*cE*cA)*mag_b
            Ly = (cE*sA)*mag_b
            Lz = (sphi*sE+cphi*cE*cA)*mag_b
            #conversion to uvw, where w points to the phase reference centre
            rotation_in_radians = deg2rad(timestamp*integration_length_in_deg + ref_ra)
            sin_ra = sin(rotation_in_radians)
            cos_ra = cos(rotation_in_radians)
            sin_dec = sin(reference_dec_rad)
            cos_dec = cos(reference_dec_rad)
            u = -sin_ra*Lx + cos_ra*Ly
            v = -sin_dec*cos_ra*Lx - sin_dec*sin_ra*Ly + cos_dec*Lz
            w = cos_dec*cos_ra*Lx + cos_dec*sin_ra*Ly + sin_dec*Lz
            self.uvw[r] = [u,v,w]

        #Create array to store V(u,v,w) during MCMC
        self.vis_tmp = zeros(row_count)

        #Get max values of u and v and set cell sizes
        max_u = max(abs(self.uvw[:,0]))
        min_u = min(abs(self.uvw[:,0]))
        max_v = max(abs(self.uvw[:,1]))
        min_v = min(abs(self.uvw[:,1]))
        print "Maximum u,v: (%f,%f)" % (max_u,max_v)
#        self.deltal = 1.0/(2*max_u + 1e-5)
#        self.deltam = 1.0/(2*max_v + 1e-5)
#        self.Nl = 1.0/(min_u*self.deltal)
#        self.Nm = 1.0/(min_v*self.deltam)
#        self.l0 = cos(reference_dec_rad)*sin()

        #N * cell_size_in_rads = 2 * max_uv so cell_size_in_rads = 2 * max_uv / N
        cell_size_u = 2 * max_u / (self.Nx)
        cell_size_v =  2 * max_v / (self.Ny)
        print "Nyquest cell size (radians) in image space (%f,%f)" % (cell_size_u,cell_size_v)

        #Get scaled uv (why is this necessary?)
        self.scaled_uv = copy(self.uvw[:,0:2])
        self.scaled_uv[:,0] /= cell_size_u
        self.scaled_uv[:,1] /= cell_size_v
        self.scaled_uv[:,0] += self.Nx/2
        self.scaled_uv[:,1] += self.Ny/2
        print "Maximum scaled u,v: (%f,%f)" % (max(self.scaled_uv[:,0]),max(self.scaled_uv[:,1]))
        if plot_on:
            plt.figure(figsize=(10, 10))
            plt.title("Sampling of the measurement space (amplitude)")
            plt.imshow(10*log10(abs(self.model_vis)+1e-10))
            plt.plot(self.scaled_uv[:,0],self.scaled_uv[:,1],"w.",label="Baselines")
            plt.colorbar()
            hrs = int(observation_length_in_hrs)
            mins = int(observation_length_in_hrs * 60 - hrs*60)
            plt.figure(figsize=(10,10))
            plt.title("UV COVERAGE (%dh:%dm @ RA=%f, DEC=%f)" % (hrs,mins,ref_ra,ref_dec))
            plt.plot(self.uvw[:,0],self.uvw[:,1],"r.",label="Baselines")
            plt.plot(-self.uvw[:,0],-self.uvw[:,1],"b.",label="Conjugate Baselines")
            plt.xlabel(r"$u / [cycles\cdot rad^{-1}\cdot m^{-1}]$",fontsize=18)
            plt.ylabel(r"$v / [cycles\cdot rad^{-1}\cdot m^{-1}]$",fontsize=18)
            plt.legend(bbox_to_anchor=(1.75, 1.0))
            plt.show()
        return

    def fft_degrid(self,Image,mode=1):
        """
        Convolutional degridder
        Input:  Image = the image to degrid
        Keyword arguments:
        scaled_uv --- interferometer's uv coordinates. (Prerequisite: these uv points are already scaled by the simularity
                      theorem, such that -N_x*Cell_l*0.5 <= theta_l <= N_x*Cell_l*0.5 and -N_y*Cell_m*0.5 <= theta_m <= N_y*Cell_m*0.5
        Nx,Ny --- size of image in pixels
        convolution_filter --- pre-instantiated anti-aliasing filter object
        """
        if mode=="set":
            self.model_vis = fft.fftshift(fft.fft2(fft.ifftshift(Image)))
            vis_grid = self.model_vis
        else:
            vis_grid = fft.fftshift(fft.fft2(fft.ifftshift(Image)))
        vis = zeros([self.scaled_uv.shape[0]],dtype=complex)
        for r in range(0,self.scaled_uv.shape[0]):
            disc_u = int(round(self.scaled_uv[r,0]))
            disc_v = int(round(self.scaled_uv[r,1]))
            frac_u_offset = int((1 - self.scaled_uv[r,0] + disc_u) * self.oversample)
            frac_v_offset = int((1 - self.scaled_uv[r,1] + disc_v) * self.oversample)
            if (disc_v + self.full_sup_wo_padding  >= self.Ny or
                disc_u + self.full_sup_wo_padding >= self.Nx or
                disc_v < 0 or disc_u < 0):
                continue
            interpolated_value = 0.0 + 0.0j
            for conv_v in range(0,self.full_sup_wo_padding):
                v_tap = self.filter_taps[conv_v * self.oversample + frac_v_offset]
                for conv_u in range(0,self.full_sup_wo_padding):
                    u_tap = self.filter_taps[conv_u * self.oversample + frac_u_offset]
                    conv_weight = v_tap * u_tap
                    interpolated_value += vis_grid[disc_u - self.half_sup + conv_u, disc_v - self.half_sup + conv_v] * conv_weight
            vis[r] = interpolated_value
        if mode=="set":
            #Generate some imperfect data
            self.vis_uv = vis
            sigma = 1.0
            self.svisdat = ones(vis.size)*(sigma*vis + 1j*sigma*vis)
            self.svisdatsq = real(self.svisdat*self.svisdat.conj()) + 1.0e-10
            self.visdat = vis #+ sigma*vis*random.randn(vis.size) + 1j*sigma*vis*random.randn(vis.size)
        return vis

    def grid_ifft(self,vis):
        """
        Convolutional gridder

        Keyword arguments:
        vis --- Visibilities as sampled by the interferometer
        scaled_uv --- interferometer's uv coordinates. (Prerequisite: these uv points are already scaled by the simularity
                      theorem, such that -N_x*Cell_l*0.5 <= theta_l <= N_x*Cell_l*0.5 and -N_y*Cell_m*0.5 <= theta_m <= N_y*Cell_m*0.5
        Nx,Ny --- size of image in pixels
        convolution_filter --- pre-instantiated AA_filter anti-aliasing filter object
        """
        measurement_regular = zeros([self.Nx,self.Ny],dtype=complex) #one grid for the resampled visibilities
        #for deconvolution the PSF should be 2x size of the image (see Hogbom CLEAN for details)
        sampling_regular = zeros([2*self.Nx,2*self.Ny],dtype=complex) #one grid for the resampled sampling function
        for r in range(0,self.scaled_uv.shape[0]):
            disc_u = int(round(self.scaled_uv[r,0]))
            disc_v = int(round(self.scaled_uv[r,1]))
            frac_u_offset = int((1 - self.scaled_uv[r,0] + disc_u) * self.oversample)
            frac_v_offset = int((1 - self.scaled_uv[r,1] + disc_v) * self.oversample)
            if (disc_v + self.full_sup_wo_padding  >= self.Ny or
                disc_u + self.full_sup_wo_padding >= self.Nx or
                disc_v < 0 or disc_u < 0):
                continue
            for conv_v in range(0,self.full_sup_wo_padding):
                v_tap = self.filter_taps[conv_v * self.oversample + frac_v_offset]
                for conv_u in range(0,self.full_sup_wo_padding):
                    u_tap = self.filter_taps[conv_u * self.oversample + frac_u_offset]
                    conv_weight = v_tap * u_tap
                    #print conv_weight
                    measurement_regular[disc_u - self.half_sup + conv_u, disc_v - self.half_sup + conv_v] += vis[r] * conv_weight
                    sampling_regular[disc_u - self.half_sup + conv_u, disc_v - self.half_sup + conv_v] += (1+0.0j) * conv_weight
        #Get dirty image and psf
        self.IDIRTY = fft.fftshift(fft.ifft2(fft.ifftshift(measurement_regular)))
        self.psf = abs(fft.fftshift(fft.ifft2(fft.ifftshift(sampling_regular))))
        self.IDIRTY = abs(self.IDIRTY/max(self.psf)) # normalize by the centre value of the PSF
        return

    def set_AA_filter(self, filter_half_support, filter_oversampling_factor, filter_type):
        self.half_sup = filter_half_support
        self.oversample = filter_oversampling_factor
        self.full_sup_wo_padding = (filter_half_support * 2 + 1)
        self.full_sup = self.full_sup_wo_padding + 2 #+ padding
        self.no_taps = self.full_sup + (self.full_sup - 1) * (filter_oversampling_factor - 1)
        taps = arange(-self.no_taps//2,self.no_taps//2 + 1)/float(filter_oversampling_factor)
        self.taps = taps
        if filter_type == "box":
            self.filter_taps = where((taps >= -0.5) & (taps <= 0.5),ones([len(taps)]),zeros([len(taps)]))
        elif filter_type == "sinc":
            self.filter_taps = sinc(taps)
        elif filter_type == "gaussian_sinc":
            alpha_1=1.55
            alpha_2=2.52
            self.filter_taps = sin(pi/alpha_1*(taps+1.0e-11))/(pi*(taps+1.0e-11))*exp(-(taps/alpha_2)**2)
        else:
            raise ValueError("Expected one of 'box','sinc' or 'gausian_sinc'")
        return

    def setup(self):
        self.NO_ANTENNA = 27
        self.NO_BASELINES = self.NO_ANTENNA * (self.NO_ANTENNA - 1) / 2 + self.NO_ANTENNA
        self.CENTRE_CHANNEL = 1.0e9 / 299792458 #Wavelength of 1 GHz
        #Antenna positions (from Measurement Set "ANTENNA" table)
        #Here we assumed these are in Earth Centred Earth Fixed coordinates, see:
        #https://en.wikipedia.org/wiki/ECEF
        #http://casa.nrao.edu/Memos/229.html#SECTION00063000000000000000
        self.ANTENNA_POSITIONS = array([[-1601710.017000 , -5042006.925200 , 3554602.355600],
                                      [-1601150.060300 , -5042000.619800 , 3554860.729400],
                                      [-1600715.950800 , -5042273.187000 , 3554668.184500],
                                      [-1601189.030140 , -5042000.493300 , 3554843.425700],
                                      [-1601614.091000 , -5042001.652900 , 3554652.509300],
                                      [-1601162.591000 , -5041828.999000 , 3555095.896400],
                                      [-1601014.462000 , -5042086.252000 , 3554800.799800],
                                      [-1601185.634945 , -5041978.156586 , 3554876.424700],
                                      [-1600951.588000 , -5042125.911000 , 3554773.012300],
                                      [-1601177.376760 , -5041925.073200 , 3554954.584100],
                                      [-1601068.790300 , -5042051.910200 , 3554824.835300],
                                      [-1600801.926000 , -5042219.366500 , 3554706.448200],
                                      [-1601155.635800 , -5041783.843800 , 3555162.374100],
                                      [-1601447.198000 , -5041992.502500 , 3554739.687600],
                                      [-1601225.255200 , -5041980.383590 , 3554855.675000],
                                      [-1601526.387300 , -5041996.840100 , 3554698.327400],
                                      [-1601139.485100 , -5041679.036800 , 3555316.533200],
                                      [-1601315.893000 , -5041985.320170 , 3554808.304600],
                                      [-1601168.786100 , -5041869.054000 , 3555036.936000],
                                      [-1601192.467800 , -5042022.856800 , 3554810.438800],
                                      [-1601173.979400 , -5041902.657700 , 3554987.517500],
                                      [-1600880.571400 , -5042170.388000 , 3554741.457400],
                                      [-1601377.009500 , -5041988.665500 , 3554776.393400],
                                      [-1601180.861480 , -5041947.453400 , 3554921.628700],
                                      [-1601265.153600 , -5041982.533050 , 3554834.858400],
                                      [-1601114.365500 , -5042023.151800 , 3554844.944000],
                                      [-1601147.940400 , -5041733.837000 , 3555235.956000]]);
        self.ARRAY_LATITUDE = 34 + 4 / 60.0 + 43.497 / 3600.0 #Equator->North
        self.ARRAY_LONGITUDE = -(107 + 37 / 60.0 + 03.819 / 3600.0) #Greenwitch->East, prime -> local meridian
        self.REF_ANTENNA = 0
        #Conversion from ECEF -> ENU:
        #http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
        slambda = sin(deg2rad(self.ARRAY_LONGITUDE))
        clambda = cos(deg2rad(self.ARRAY_LONGITUDE))
        sphi = sin(self.ARRAY_LONGITUDE)
        cphi = cos(self.ARRAY_LATITUDE)
        ecef_to_enu = [[-slambda,clambda,0],
                       [-clambda*sphi,-slambda*sphi,cphi],
                       [clambda*cphi,slambda*cphi,sphi]]
        self.ENU = empty(self.ANTENNA_POSITIONS.shape)
        for a in range(0,self.NO_ANTENNA):
            self.ENU[a,:] = dot(ecef_to_enu,self.ANTENNA_POSITIONS[a,:])
        self.ENU -= self.ENU[self.REF_ANTENNA]
        return

    def set_max_lik_slow(self,beta,loglikslow,fmean):
        self.maxlikslow = loglikslow

        self.best_beta = beta
        self.best_fmean = fmean
        return

    def set_max_lik(self,Image,loglik):
        self.maxlik = loglik
        self.best_Image = Image
        return

    def track_max_lik_slow(self,beta,loglikslow,fmean):
        if loglikslow < self.maxlik:
            self.maxlikslow = loglikslow
            self.best_beta = beta
            self.best_fmean = fmean
        return

    def track_max_lik(self,Image,loglik):
        if loglik < self.maxlik:
            self.maxlik = loglik
            self.best_Image = Image
            print 'Max lik tracked'
        return

    def MCMC_slow_step(self,beta0,loglikslow0):
        """
        Here we implement the slow MCMC step
        """
        #Propose new beta (fix this to reject when any beta<0)
        beta = beta0 + random.multivariate_normal(zeros(2),self.SIGMAslow*eye(2))
        if (beta<0.0).any():
            return beta0,loglikslow0,0
        else:
            #Get updated fmean (note fcov does not depend on beta)
            fmean = self.get_meanf(beta)
            #Compute the likelihood with the mean function
            vis = self.fft_degrid(fmean.reshape((self.Nx,self.Ny)),mode=1)
            loglikslow = self.get_chi2(vis)
            logr = loglikslow-loglikslow0
            accprob = exp(-logr)
            u = random.random()
            if u < accprob:#Accept
                #Update the mean function to use accepted value of beta
                self.fmean = fmean
                #track the maximum likelihood
                self.track_max_lik_slow(beta,loglikslow,fmean)
                return beta, loglikslow, 1
            else:
                #Reject the sample
                return beta0,loglikslow0,0

    def MCMC_step(self,Image0,loglik0):
        """
        Here we implement the fast MCMC step
        """
        Image_flat = self.sample(Image0.flatten())
        if (Image_flat<0.0).any():
            I = argwhere(Image_flat<0)
            Image_flat[I] = 0.0
            #return Image0, loglik0, 0
        #else:
        Image = Image_flat.reshape(self.Nx,self.Ny)
        vis = self.fft_degrid(Image,mode=1)
        loglik = self.get_chi2(vis)
        logr = loglik-loglik0
        accprob = exp(-logr)
        u = random.random()
        if u < accprob:
            #Accept the sample and track the maximum likelihood
            self.track_max_lik(Image,loglik)
            return Image, loglik, 1
        else:
            if (Image0<0.0).any():
                I = argwhere(Image0<0)
                Image0[I] = 0.0
                print "We should never get here"
            #Reject the sample
            return Image0,loglik0,0

    def save_and_plot(self,SAVE=True,PLOT=True):
        if SAVE:
            #Save everything
            savez("Sky_Prior.npz",fmean=self.fmean,fmeanNP = self.fmeanNP,RP = self.RP,beta=self.beta,fcov=self.fcov,W=self.W,V=self.V,model_sky=self.model_sky,ICLEAN=self.ICLEAN,visdat=self.visdat,svisdatsq=self.svisdatsq,vis_uv=self.vis_uv,scaled_uv=self.scaled_uv)
        if PLOT:
            plt.figure(figsize=(25, 15))
            plt.subplot(221)
            plt.title("Model sky")
            plt.imshow(self.model_sky,cmap="gray")
            plt.colorbar()
            plt.xlabel(r"$l$",fontsize=18)
            plt.ylabel(r"$m$",fontsize=18)
            plt.subplot(222)
            plt.title("ICLEAN")
            plt.imshow(self.ICLEAN,cmap="gray")
            plt.colorbar()
            plt.xlabel(r"$l$",fontsize=18)
            plt.ylabel(r"$m$",fontsize=18)
            plt.subplot(223)
            plt.title("Dirty map")
            plt.imshow(self.IDIRTY,cmap="gray")
            plt.colorbar()
            plt.xlabel(r"$l$",fontsize=18)
            plt.ylabel(r"$m$",fontsize=18)
            plt.subplot(224)
            plt.title("ICLEAN2")
            plt.imshow(self.IPREDICT,cmap="gray")
            plt.colorbar()
            plt.xlabel(r"$l$",fontsize=18)
            plt.ylabel(r"$m$",fontsize=18)
            plt.tight_layout(pad=3.0)
            plt.show()
#            plt.figure()
#            plt.imshow(self.fmean.reshape((self.Nx,self.Ny)),cmap='gray')
#            plt.colorbar()

if __name__ == "__main__":
    #Simulate or load data model and prior
    npix = 64
    nsource = 20
    max_I = 5
    ref_ra = 0.0 #In degrees
    ref_dec = 90.0 #In degrees
    observation_length_in_hrs = 1.5
    integration_length = 60/3600.0
    filter_half_support = 3
    filter_oversampling_factor = 30
    M = python_imager(npix,nsource,max_I,filter_half_support,filter_oversampling_factor,ref_ra,ref_dec,observation_length_in_hrs,integration_length,mode='Full')

    #Start by optimising betas
    bnds = ((1e-8, None), (1e-8, None))
    beta0 = M.beta
    betap = opt.minimize(M.lik_as_func_of_beta,beta0,method='L-BFGS-B',bounds=bnds)
    beta = betap.x
    if betap['success']:
        print "Optimised betas = ", beta


    #Set MCMC params
    npar = 2 #Number of parameters in model
    nsampslow = 200 #Number of slow steps
    nadaptslow = 25
    nsampfast = 100 #Number of fast steps
    nburnslow = 25 #Length of burn in period
    nburnfast = 10 #0.05*nburnslow

    #Set starting values
    fmean = M.get_meanf(beta)
    M.fmean = fmean
    Image = fmean.reshape(npix,npix)
    visslow = M.fft_degrid(Image)
    loglikslow = M.get_chi2(visslow)
    loglik = loglikslow

    #Initialise the trackers
    M.set_max_lik_slow(beta,loglikslow,fmean)
    M.set_max_lik(Image,loglik)

    #Do burnin period
    print "Burning"
    for i in xrange(nburnslow):
        beta,loglikslow,aslow = M.MCMC_slow_step(beta,loglikslow)
        for j in xrange(nburnfast):
            Image,loglik,a = M.MCMC_step(Image,loglik)

#    #Fit the GP again (since we have a latent variable model this should be done iteratively I think)
#    M.set_prior(M.best_Image.copy(),train_theta=True)
    
    #Initialise array storage
    beta_store = zeros([nsampslow,npar]) #to store samples of parameters
    Image_store = zeros([nsampslow*nsampfast,npix,npix]) #to store samples of image
    accrateslow = zeros(2)
    accrate = zeros(2)

    #Do the MCMC
    asumslow = 0.0#0.5*nadaptslow
    asumfast = 0.5*nsampfast
    for j in xrange(nsampslow):
        #Adapt acceptance rate of slow cycle
        if (j%nadaptslow==0 and j!= 0):
            if asumslow < 0.25*nadaptslow:
                M.SIGMAslow *= 0.5
            elif asumslow > 0.75*nadaptslow:
                M.SIGMAslow *= 2.0
        print "Sampling slow cycle",j
        beta,loglikslow,aslow = M.MCMC_slow_step(beta,loglikslow)
        beta_store[j,:] = beta
        accrateslow += array([aslow,1])
        asumslow += aslow
        #Adapt acceptance rate of fastcycle
        if asumfast < 0.25*nsampfast:
            M.alpha *= 0.75
        elif asumfast > 0.65*nsampfast and M.alpha < 0.75:
            M.alpha *= 1.25
        asumfast = 0
        for i in xrange(nsampfast):
            if i%10==0:
                print i, loglik
            Image,loglik,a = M.MCMC_step(Image,loglik)
            Image_store[i,:,:] = Image
            accrate += array([a,1])
            asumfast += a

    print "Acc rate slow = ",accrateslow[0]/accrateslow[1]
    print "Acceptance rate = ",accrate[0]/accrate[1]

    savez("Samples2.npz",Images=Image_store,best_Image=M.best_Image)
    #Show the best image
    plt.figure("Best")
    plt.imshow(M.best_Image,cmap='gray')
    plt.colorbar()
    plt.figure('Model')
    plt.imshow(M.model_sky,cmap='gray')
    plt.colorbar()
    plt.figure('CLEAN')
    plt.imshow(M.ICLEAN,cmap='gray')
    plt.colorbar()

    #Print some diagnostics to see if the image actually improved
    #Get residuals betweeen model and CLEAN and model and best_Image
    res1 = (M.model_sky - M.ICLEAN)**2
    res2 = (M.model_sky - M.best_Image)**2
    print "Residual of CLEAN = ", sum(res1)
    print "Residual of Best = ", sum(res2)

    #Compare the chisquared values (the smallest one wins)
    vis1 = M.fft_degrid(M.ICLEAN,mode=1)
    chi21 = M.get_chi2(vis1)
    vis2 = M.fft_degrid(M.best_Image,mode=1)
    chi22 = M.get_chi2(vis2)
    #chi2Mod = M.get_chi2(M.vis_uv)
    print "Chisq of CLEAN = ", chi21
    print "Chisq of Best = ", chi22
    #print "Chisq of Model = ", chi2Mod

    #Report best beta values
    print "Best beta = ", M.best_beta