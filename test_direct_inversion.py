# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:15:22 2017

@author: landman

direct inversion in terms of basis funcs using DFT to predict 

"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.integrate import simps
from GP.Class2DRRGP import RR_2DGP as GP
from pyrap.tables import table as tbl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def diag_dot(A,B):
    """
    Computes the diagonal of C = AB where A and B are square matrices
    """
    D = np.zeros(A.shape[0])
    for i in xrange(A.shape[0]):
        D[i] = np.dot(A[i,:],B[:,i])
    return D

def basis_func(l,m,i,j,L):
    return np.sin(np.pi*i*(l + L)/(2*L))*np.sin(np.pi*j*(m+L)/(2*L))/L

def radec_to_lm(ra,dec):
    delta_ra = ra - ra0
    l = (np.cos(dec)*np.sin(delta_ra))
    m = (np.sin(dec)*np.cos(dec0) - np.cos(dec)*np.sin(dec0)*np.cos(delta_ra))
    return l,m
    
def give_ra_dec(ra0,dec0,delta_pix,Npix):
    ra = ra0 + np.linspace(ra0 - Npix*delta_pix/2.0,ra0 + Npix*delta_pix/2.0,Npix)
    dec = dec0 + np.linspace(dec0 - Npix*delta_pix/2.0,dec0 + Npix*delta_pix/2.0,Npix)
    return ra, dec

def make_lmn_vec(l,m,Npix):
    ncoord = np.zeros(Npix*Npix)
    lmn = np.zeros([Npix*Npix,3])
    for i in xrange(Npix):
       for j in xrange(Npix):
           ncoord[i*Npix + j] = np.sqrt(1.0 - l[i]**2 - m[j]**2)
           lmn[i*Npix + j,0] = l[i]
           lmn[i*Npix + j,1] = m[j]
           lmn[i*Npix + j,2] = ncoord[i*Npix+j] - 1.0 
    return lmn, ncoord
    
if __name__=="__main__":
    # Load in some data
    ms = tbl("/home/landman/Projects/algorithm_notebooks/DATA/mfs_sky.MS_p0/")
    #ms = tbl("DATA/Challenge_1.ms/")

    #Get freq info
    msfreq = tbl("/home/landman/Projects/algorithm_notebooks/DATA/mfs_sky.MS_p0::SPECTRAL_WINDOW")
    #msfreq = tbl("DATA/Challenge_1.ms::SPECTRAL_WINDOW")

    #Get pointing center (to compute DFT kernel)
    msfield = tbl("/home/landman/Projects/algorithm_notebooks/DATA/mfs_sky.MS_p0::FIELD")
    #msfield = tbl("DATA/Challenge_1.ms::FIELD")

    ra0, dec0 = msfield.getcol("PHASE_DIR").squeeze() #in radians
    #print ra0, dec0

    c = 2.99792458e8 #speed of light

    uvw = ms.getcol("UVW")[:,:]
    print uvw.shape
    vis = ms.getcol("DATA")[:,0,:]
    weights = ms.getcol("WEIGHT")

    Nrows = uvw.shape[0]

    Freqs = msfreq.getcol("CHAN_FREQ").squeeze()[0]
    nchan = Freqs.size
    ref_freq = Freqs

    #Close tables
    ms.close()
    msfreq.close()
    msfield.close()

    #Get wavelengths
    ref_lambda = c / Freqs

    #Instantiate the gridder
    #Get V corresponding to stokes I
    Vobs = (vis[:,0] + vis[:,3])*0.5 #/weights[:,0]
    #Vobs = (vis[:,0] + vis[:,3])*0.5/weights[:,0]


    # make image domain vector
    # ARCSEC2RAD = 4.8481e-6
    # delta_pix = ARCSEC2RAD * Npix
    # uv_scale = Npix * delta_pix
    # scaled_uvw = uvw * uv_scale
    # ra,dec = give_ra_dec(ra0,dec0,delta_pix,Npix)
    # l,m = radec_to_lm(ra,dec)
    # Set max length
    Npix = 129  # note using odd number otherwise centre of PSF is not well defined
    L = 0.015

    l = np.linspace(-L,L,Npix)
    m = np.linspace(-L,L,Npix)
    ll, mm = np.meshgrid(l, m)
    lmn, ncoord = make_lmn_vec(l, m, Npix)
    lm = lmn[:,0:2].T
    print lmn.shape

    # Set number of basis funcs
    Nbasis = 15
    
    # Compute the matrix that maps coefficients to visibilities
    Iint = np.zeros([Nrows, Nbasis**2], dtype=np.complex)
    H = np.zeros([Npix**2, Nbasis**2])
    print "Computing coeffs to vis operator mapping"
    for k in xrange(Nrows):
        K = np.exp(-2.0j*np.pi*np.dot(uvw[k,:],lmn.T)/ref_lambda)/np.sqrt(Nrows)
        if k % 1000 == 0:
            print k*100/Nrows, "percent done"
        for i in xrange(Nbasis):
            for j in xrange(Nbasis):
                H[:, i * Nbasis + j] = basis_func(lmn[:, 0], lmn[:, 1], i + 1, j + 1, L)/ncoord
                Iint[k, i * Nbasis + j] = np.dot(K, H[:, i * Nbasis + j])

    # Show matrix rank and conditioning number
    print "rank = ", np.linalg.matrix_rank(Iint)
    print "cond = ", np.linalg.cond(Iint)

    # Save Iint to file so we don't have to recompute it every time
    print "Saving"
    np.savez('/home/landman/Projects/My_Imagers/Iint' + str(Npix) + 'pix' + str(Nbasis) + 'bf.npz', Iint=Iint)
    
    # # Get the dirty image and PSF
    # Idflat = np.zeros(Npix**2)
    # PSFflat = np.zeros(Npix**2)
    # Onesflat = np.ones(Nrows, dtype=np.complex)
    # for n in xrange(Npix**2):
    #     Kinv = np.exp(2.0j*np.pi*np.dot(lmn[n, :], uvw.T))/np.sqrt(Nrows)
    #     Idflat[n] = np.dot(Kinv, Vobs).real
    #     PSFflat[n] = np.dot(Kinv, Onesflat).real
    #     if n%250 == 0:
    #         print "n = ", n
    #
    # # plot DFT result
    # PSFmax = PSFflat.max()
    # Idflat /= PSFmax
    # plt.figure('DFT Id')
    # plt.imshow(Idflat.reshape(Npix, Npix), interpolation="nearest", cmap="cubehelix")
    # plt.colorbar()
    # plt.savefig("/home/landman/Projects/My_Imagers/figures/ID_real" + str(Npix) + ".png", dpi=250)
    #
    # # plot DFT result
    # PSFflat /= PSFmax
    # plt.figure('DFT PSF')
    # plt.imshow(PSFflat.reshape(Npix, Npix), interpolation="nearest", cmap="cubehelix")
    # plt.colorbar()
    # plt.savefig("/home/landman/Projects/My_Imagers/figures/PSF_real" + str(Npix) + ".png", dpi=250)

    # np.savez("ID_and_PSF_" + str(Npix) + "pix.npz", ID=Idflat.reshape(Npix,Npix), PSF=PSFflat.reshape(Npix,Npix), PSFmax=PSFmax)

    # holder = np.load("ID_and_PSF_" + str(Npix) + "pix.npz")
    # PSF = holder["PSF"]
    # #PSF = PSFflat.reshape(Npix, Npix)
    # PSFflat = PSF.flatten()
    # ID = holder["ID"]
    # Idflat = ID.flatten()
    # PSFmax = holder["PSFmax"]
    #
    # print "PSFmax = ", PSFmax
    #
    # PSFarea1 = simps(PSF, m)
    # PSFarea2 = simps(PSFarea1, l)
    #
    # PSFnorm = PSF/PSFarea2
    #
    # test = simps(simps(PSFnorm, l), m)
    #
    # PSFsum = np.sum(PSFflat)
    #
    # print PSFsum
    #
    # # Do the GPR
    # GPR = GP(lm, lm, L, Nbasis, 2)
    #
    # print "Convolving basis funcs"
    # #GPR.RR_convolve_basis(PSFflat.reshape(Npix, Npix), ll, mm, L)
    # GPR.RR_convolve_basis(PSF/PSFsum, ll, mm, L)
    #
    # theta = np.array([0.01, 0.1*L, 1.0])
    #
    # print "training GP"
    # coeffs, theta = GPR.RR_EvalGP_conv(theta, Idflat)
    #
    # print "Done"
    #
    # #Icoeffs = np.argwhere(coeffs < 1e-3)
    # #coeffs2 = coeffs
    # #coeffs2[Icoeffs] = 0.0
    #
    # #print coeffs2.size
    #
    # print theta
    #
    # IdGP = GPR.RR_From_Coeffs(coeffs)
    #
    # #Iltz = np.argwhere(IdGP < 0.1*IdGP.max()).squeeze()
    # #IdGP[Iltz] = 0.0
    #
    # fcovcoeffs = GPR.RR_covf_conv(theta, return_covf=False)
    #
    # # Draw a sample
    # coeffssamp = np.random.multivariate_normal(coeffs, fcovcoeffs)
    #
    # Idsamp = GPR.RR_From_Coeffs(coeffssamp)
    #
    # # Plot result
    # plt.figure('Id4')
    # plt.imshow(Idsamp.reshape(Npix,Npix), interpolation="nearest", cmap="cubehelix")
    # plt.colorbar()
    # plt.savefig("/home/landman/Projects/My_Imagers/figures/IM_GP_samp" + str(Npix) + ".png", dpi=250)
    #
    # # Plot result
    # plt.figure('Id3')
    # plt.imshow(IdGP.reshape(Npix,Npix), interpolation="nearest", cmap="cubehelix")
    # plt.colorbar()
    # plt.savefig("/home/landman/Projects/My_Imagers/figures/IM_GP" + str(Npix) + ".png", dpi=250)

