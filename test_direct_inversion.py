#!/usr/bin/env python
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
import PyPolyChord as PolyChord

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


def twoD_Gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    return amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))

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

    Freqs = msfreq.getcol("CHAN_FREQ").squeeze()
    #print Freqs
    Freqs = Freqs[0]

    nchan = Freqs.size
    ref_freq = Freqs

    #Close tables
    ms.close()
    msfreq.close()
    msfield.close()

    #Get wavelengths
    ref_lambda = c / Freqs
    print "ref lambda = ", ref_lambda



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
    Npix = 257  # note using odd number otherwise centre of PSF is not well defined
    L = 0.003

    l = np.linspace(-L,L,Npix)
    m = np.linspace(-L,L,Npix)
    ll, mm = np.meshgrid(l, m)
    lmn, ncoord = make_lmn_vec(l, m, Npix)
    lm = lmn[:,0:2].T

    print lmn.shape

    # Get and plot true model image
    IM = np.zeros([Npix, Npix])
    sigma_x = 0.025 * np.pi / (180.0)  # Convert to radians (the factor of 2 is to match the tigger parametrisation)
    sigma_y = 0.025 * np.pi / (180.0)
    x0 = -0.05 * np.pi / (180.0)
    y0 = -0.05 * np.pi / (180.0)
    theta = 0.0
    IM += twoD_Gaussian(ll, mm, 100, x0, y0, sigma_x, sigma_y, theta)
    sigma_x = 0.015 * np.pi / (180.0)  # Convert to radians (the factor of 2 is to match the tigger parametrisation)
    sigma_y = 0.015 * np.pi / (180.0)
    x0 = 0.05 * np.pi / (180.0)
    y0 = 0.05 * np.pi / (180.0)
    theta = 0.0
    IM += twoD_Gaussian(ll, mm, 50, x0, y0, sigma_x, sigma_y, theta)

    IM = np.flipud(np.fliplr(IM))

    # plot IM
    plt.figure('IM true')
    plt.imshow(IM, interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/IM_true" + str(Npix) + ".png", dpi=250)

    # Print Nyquist criteria
    deltapix = l[1] - l[0]
    umax = np.abs(uvw[:,0]).max()/ref_lambda
    umin = np.abs(uvw[:,0]).min()/ref_lambda
    vmax = np.abs(uvw[:,1]).max()/ref_lambda
    vmin = np.abs(uvw[:,1]).min()/ref_lambda

    print "delta pix should be less than", 1.0/(2*np.maximum(umax,vmax)), " whereas you are at ", deltapix
    print "Ndeltapix should be more than", 1.0/np.minimum(umin,vmin), " whereas you are at", deltapix*Npix

    # Set number of basis funcs
    Nbasis = 9

    try:
        Iint = np.load('/home/landman/Projects/My_Imagers/Iint' + str(Npix) + 'pix' + str(Nbasis) + 'bf.npz')["Iint"]
    except:
        # Compute the matrix that maps coefficients to visibilities
        Iint = np.zeros([Nrows, Nbasis**2], dtype=np.complex)
        H = np.zeros([Npix**2, Nbasis**2])
        print "Computing coeffs to vis operator mapping"
        for k in xrange(Nrows):
            K = np.exp(-2.0j*np.pi*np.dot(uvw[k,:],lmn.T)/ref_lambda)/np.sqrt(Nrows)
            #K = np.exp(-2.0j * np.pi * (uvw[k, 0]*lmn[:,0] + uvw[k, 1]*lmn[:,1] + uvw[k, 1]*lmn[:,1]) / ref_lambda) / np.sqrt(Nrows)
            if k % 1000 == 0:
                print k*100/Nrows, "percent done"
            for i in xrange(Nbasis):
                for j in xrange(Nbasis):
                    H[:, i * Nbasis + j] = basis_func(lmn[:, 0], lmn[:, 1], i + 1, j + 1, L)/ncoord
                    Iint[k, i * Nbasis + j] = np.dot(K, H[:, i * Nbasis + j])
                    #Iint[k, i * Nbasis + j] = np.dot(K, H[:, i * Nbasis + j])

        # Show matrix rank and conditioning number
        print "rank = ", np.linalg.matrix_rank(Iint)
        print "cond = ", np.linalg.cond(Iint)

        # Save Iint to file so we don't have to recompute it every time
        print "Saving"
        np.savez('/home/landman/Projects/My_Imagers/Iint' + str(Npix) + 'pix' + str(Nbasis) + 'bf.npz', Iint=Iint)

    # Get the dirty image and PSF
    try:
        holder = np.load("ID_and_PSF_" + str(Npix) + "pix.npz")
        PSFmax = holder["PSFmax"]
        PSF = holder["PSF"]  # *PSFmax
        # PSF = PSFflat.reshape(Npix, Npix)
        PSFflat = PSF.flatten()
        ID = holder["ID"]  # *PSFmax
        Idflat = ID.flatten()
    except:
        Idflat = np.zeros(Npix**2)
        PSFflat = np.zeros(Npix**2)
        Onesflat = np.ones(Nrows, dtype=np.complex)
        for n in xrange(Npix**2):
            Kinv = np.exp(2.0j*np.pi*np.dot(lmn[n, :], uvw.T)/ref_lambda)/np.sqrt(Nrows)
            Idflat[n] = np.dot(Kinv, Vobs).real
            PSFflat[n] = np.dot(Kinv, Onesflat).real
            if n%250 == 0:
                print "n = ", n

        # plot DFT result
        PSFmax = PSFflat.max()
        Idflat /= PSFmax
        plt.figure('DFT Id')
        plt.imshow(Idflat.reshape(Npix, Npix), interpolation="nearest", cmap="cubehelix")
        plt.colorbar()
        plt.savefig("/home/landman/Projects/My_Imagers/figures/ID_real" + str(Npix) + ".png", dpi=250)

        # plot DFT result
        PSFflat /= PSFmax
        plt.figure('DFT PSF')
        plt.imshow(PSFflat.reshape(Npix, Npix), interpolation="nearest", cmap="cubehelix")
        plt.colorbar()
        plt.savefig("/home/landman/Projects/My_Imagers/figures/PSF_real" + str(Npix) + ".png", dpi=250)

        np.savez("ID_and_PSF_" + str(Npix) + "pix.npz", ID=Idflat.reshape(Npix,Npix), PSF=PSFflat.reshape(Npix,Npix), PSFmax=PSFmax)


    print "PSFmax = ", PSFmax

    # PSFarea1 = simps(PSF, m)
    # PSFarea2 = simps(PSFarea1, l)
    #
    # PSFnorm = PSF/PSFarea2
    #
    # test = simps(simps(PSFnorm, l), m)

    PSFsum = np.abs(np.sum(PSFflat))

    print "PSF sum = ", PSFsum

    # Do the GPR
    GPR = GP(lm, lm, L, Nbasis, 2)

    print "Convolving basis funcs"
    #GPR.RR_convolve_basis(PSFflat.reshape(Npix, Npix), ll, mm, L)
    GPR.RR_convolve_basis(PSF/PSFsum, ll, mm, L)

    print "Training dirty GP"
    theta = np.array([0.01, 0.1*L, 1.0])
    coeffs, theta = GPR.RR_EvalGP_conv(theta, Idflat)

    print "Done. theta = ", theta

    # Use coeffs to reconstruct initial "clean" estimate
    IdGP = GPR.RR_From_Coeffs(coeffs)

    # Get the covariance matrix of the parameters
    fcovcoeffs = GPR.RR_covf_conv(theta, return_covf=False)

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
    # Plot result
    plt.figure('IdGP')
    plt.imshow(IdGP.reshape(Npix,Npix), interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/IM_GP" + str(Npix) + 'pix' + str(Nbasis) + 'bf.png', dpi=250)


    # Map Coeffs to visibilities
    Vpred = np.dot(Iint, coeffs)

    # get the Chi2
    Vdiff = Vobs - Vpred
    Chi2real = np.sum(Vdiff.real**2)
    Chi2imag = np.sum(Vdiff.imag**2)

    print "Chi2 = ", Chi2imag + Chi2real

    # # Produce dirty image with these visibilities
    # Idpred = np.zeros(Npix**2)
    # for n in xrange(Npix**2):
    #     Kinv = np.exp(2.0j*np.pi*np.dot(lmn[n, :], uvw.T) / ref_lambda)/np.sqrt(Nrows)
    #     Idpred[n] = np.dot(Kinv, Vpred).real
    #     if n%250 == 0:
    #         print "n = ", n
    #
    # # Plot result
    # Idpred /= PSFmax
    # plt.figure('Idpred')
    # plt.imshow(Idpred.reshape(Npix, Npix), interpolation="nearest", cmap="cubehelix")
    # plt.colorbar()
    # plt.savefig("/home/landman/Projects/My_Imagers/figures/ID_pred" + str(Npix) + ".png", dpi=250)

    # Do GPR on model image
    coeffsIM, thetaIM = GPR.RR_EvalGP(theta, IM.flatten())

    IMp = GPR.RR_From_Coeffs(coeffsIM).reshape(Npix, Npix)

    # Plot result
    plt.figure('IMp_GP')
    plt.imshow(IMp, interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/IMp_GP" + str(Npix) + 'pix' + str(Nbasis) + 'bf.png', dpi=250)

    # Test PolyChord
    nDims = coeffs.size
    nDerived = 0

    # Check if prior is adequate
    diffcoeffs = np.abs(coeffs - coeffsIM)
    diagcov = np.sqrt(np.diag(fcovcoeffs + 0.001*np.eye(Nbasis**2)*theta[2]))

    print "Max = ", (diffcoeffs/diagcov).max(), "Mean = ", np.mean(diffcoeffs/diagcov), " Min = ", (diffcoeffs/diagcov).min()

    L = np.linalg.cholesky(fcovcoeffs + 0.001*np.eye(Nbasis**2)*theta[2])

    def simple_prior(cube):
        return coeffs + np.dot(L, cube)

    def simple_lik(theta):
        phi = [0.0] * nDerived

        V = np.dot(Iint, coeffs)
        nDims = len(theta)

        # get the Chi2
        Vdiff = Vobs - V
        Chi2real = np.sum(Vdiff.real ** 2)/2.0
        Chi2imag = np.sum(Vdiff.imag ** 2)/2.0

        logL = Chi2real + Chi2imag
        return logL, phi



    PolyChord.run_nested_sampling(simple_lik, nDims, nDerived, prior=simple_prior, num_repeats=nDims, base_dir="/home/landman/Software/PolyChord/chains")


