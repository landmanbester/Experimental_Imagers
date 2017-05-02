#!/usr/bin/env python
"""
Created on Mon Feb  6 16:15:22 2017

@author: landman

direct inversion in terms of basis funcs using DFT to predict 

"""

import numpy as np
from GP.Class2DRRGP import RR_2DGP
from scipy.signal import fftconvolve
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

#def func(x,y):
#    return 5*np.exp(-(x**2 + y**2)/(2*3))

def func(x,y):
    return 5*np.exp(-(x**2 + y**2)/(2*3))*np.sin(np.sqrt(x**2 + y**2))

def map_index(row_ind, col_ind, Np):
    """
    Returns the index corresponding to (row_ind,col_ind) in a flattened (Npix,Npix) array
    """
    return row_ind*Np + col_ind
    
if __name__=="__main__":
    # Set some params
    Nrows = 1000 #the number of points to sample in the uv plane
    Np = 257 #the number of pixels in the image
    L = 10 # Domain boundary    
    Nbasis = 20 # Number of basis funcs
    D = 2
    
    # Set image on regular grid
    l = np.linspace(-10,10,Np)
    dell = l[1]-l[0]
    ll, mm = np.meshgrid(l,l)
    lm = np.vstack((ll.flatten(),mm.flatten()))

    #Get function value with some noise added
    z = func(lm[0,:], lm[1,:]) # + 0.5*np.random.randn(Np**2)    
    
    # Get exact surface
    Im = func(ll, mm)
    # Plot result
    plt.figure('Im')
    plt.imshow(Im, interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/Direct_Inversion_IM.png", dpi=250)

    Imflat = Im.flatten()

    # Get the FFT    
    Visgrid = np.fft.fft2(Im)

    # Get grid of frequencies
    kk, jj = np.meshgrid(np.arange(Np),np.arange(Np))
    ugrid = kk/(Np*dell)
    vgrid = jj/(Np*dell)
    
    # randomly select some uv-samples
    Indx = np.random.randint(0, Np, Nrows)  # Samples along row dimension
    Indy = np.random.randint(0, Np, Nrows)  # Samples along column dimension
    
    # Find unique entries
    Icomb = np.vstack((Indx,Indy)).reshape(Nrows, 2)
    b = np.ascontiguousarray(Icomb).view(np.dtype((np.void, Icomb.dtype.itemsize * Icomb.shape[1])))
    _, idx = np.unique(b, return_index=True)
    Icomb = Icomb[idx]
    Indx = Icomb[:,0]
    Indy = Icomb[:,1]
    
    Nrows = Indx.size
    
    # Get the indices that would correspond to the flattended image
    Indflat = map_index(Indx, Indy, Np)
    
    # Get the dirty image
    Vissamps = np.zeros([Np, Np], dtype=np.complex)
    Vissamps[Indx, Indy] = Visgrid[Indx, Indy]

    # get u and v
    u = ugrid[Indx, Indy].reshape(Nrows, 1)
    v = vgrid[Indx, Indy].reshape(Nrows, 1)
    uv = np.hstack((u,v))    
    
    
    # Find DFT of basis funcs
    #Iint = np.zeros([Nrows, Nbasis**2], dtype=np.complex)
    
    Visflat = np.zeros(Nrows, dtype=np.complex)
    for k in xrange(Nrows):
        K = np.exp(-2.0j*np.pi*np.dot(uv[k,:],lm))/np.sqrt(Nrows)
        Visflat[k] = np.dot(K, Imflat) 


#    # Show matrix rank and conditioning number
#    print "rank = ",np.linalg.matrix_rank(Iint)
#    print "cond = ",np.linalg.cond(Iint)

#%%
    # draw the sampling pattern
    uv_cov = np.zeros([Np,Np])
    for i in xrange(Nrows):
        uv_cov[Indx[i],Indy[i]] += 1.0
        
    # plot DFT result
    plt.figure('uv-cov')
    plt.imshow(uv_cov, interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/uv_cov.png",dpi=250)
#%%
    # Get the usual dirty image for comparison
    Idflat = np.zeros(Np**2)
    PSFflat = np.zeros(Np**2)
    Onesflat = np.ones(Nrows, dtype=np.complex)
    for n in xrange(Np**2):
        Kinv = np.exp(2.0j*np.pi*np.dot(lm[:,n],uv.T))/np.sqrt(Nrows)
        Idflat[n] = np.dot(Kinv, Visflat).real
        PSFflat[n] = np.dot(Kinv, Onesflat).real

    # plot DFT result
    plt.figure('DFT Id')
    plt.imshow(Idflat.reshape(Np,Np), interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/ID_DFT.png", dpi=250)

    # plot DFT result
    PSFflat /= PSFflat.max()
    plt.figure('DFT PSF')
    plt.imshow(PSFflat.reshape(Np,Np), interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/PSF_DFT.png", dpi=250)

    # Get ideal PSF
    PSFideal = np.sinc(ll) * np.sinc(mm)
    plt.figure('Ideal PSF')
    plt.imshow(PSFideal, interpolation="nearest", cmap="cubehelix")
    plt.colorbar()
    plt.savefig("/home/landman/Projects/My_Imagers/figures/PSF_ideal.png", dpi=250)

#     # %%
#     H = np.zeros([Np ** 2, Nbasis ** 2])
#     Hconv = np.zeros([Np ** 2, Nbasis ** 2])
#     PSFideal = np.sinc(ll)*np.sinc(mm)
#     for i in xrange(Nbasis):
#         for j in xrange(Nbasis):
#             # Evaluate the basis funcs on lm grid
#             H[:, i * Nbasis + j] = basis_func(lm[0, :], lm[1, :], i + 1, j + 1, L)
#             #Hconv[:, i*Nbasis + j] = fftconvolve(basis_func(ll, mm, i + 1, j + 1, L), PSFideal, mode='same').flatten()
#             # Compute DFT of basis func
#             # Iint[k, i*Nbasis + j] = np.dot(K, H[:,i*Nbasis + j])
#             # %%
#
# #    # Get iFFT for comparison
# #    Id = np.fft.ifft2(Vissamps).real
# #
# #    plt.figure('FFT Id')
# #    plt.imshow(Id, interpolation="nearest", cmap="cubehelix")
# #    plt.colorbar()
#
#     # try find pinv manually
# #    HTH = diag_dot(Iint.conj().T,Iint)
# #    c = (np.diag(1.0/HTH)).dot(np.dot(Iint.conj().T,Visflat))
# #%%
#     # HTHconv = np.dot(H.T, H)
#     # Indconv = np.argwhere(HTHconv > 1e-8)
#     HTH = np.diag(np.dot(H.T, H))
#     HTHinv = np.diag(1.0/HTH)
#     #c = HTHinv.dot(np.dot(Iint.conj().T,Visflat))
#     c = HTHinv.dot(np.dot(H.T,Idflat))
#
#     # Get reconstructed image
#     Id2 = np.dot(H,c).real.reshape(Np,Np)
#
#     # Plot result
#     plt.figure('Id2')
#     plt.imshow(Id2, interpolation="nearest", cmap="cubehelix")
#     plt.colorbar()
#     plt.savefig("/home/landman/Projects/My_Imagers/figures/Direct_Inversion_ID.png", dpi=250)
#
#     # Fit a RR GP to the dirty image
#     GP = RR_2DGP(lm, lm, L, Nbasis, D)
#
#     theta0 = np.array([1.5, 0.96 , 0.3])
#
#     coeffs, theta = GP.RR_EvalGP(theta0, Idflat)
#
#     covcoeffs = GP.RR_covf(theta, return_covf=False)
#
#     Id3 = GP.RR_From_Coeffs(coeffs)
#
#     # Plot result
#     plt.figure('Id3')
#     plt.imshow(Id3.reshape(Np,Np), interpolation="nearest", cmap="cubehelix")
#     plt.colorbar()
#     plt.savefig("/home/landman/Projects/My_Imagers/figures/RR_GPR_ID.png", dpi=250)
#
#     print np.max(coeffs - c), theta
#
#     # Sample some coeffs
#     csamps = np.random.multivariate_normal(coeffs, covcoeffs)
#
#     Id4 = GP.RR_From_Coeffs(csamps)
#
#     plt.figure('Id4')
#     plt.imshow(Id4.reshape(Np,Np), interpolation="nearest", cmap="cubehelix")
#     plt.colorbar()
#     plt.savefig("/home/landman/Projects/My_Imagers/figures/RR_GPR_Sample_ID.png", dpi=250)



#%%    
#    #Plot reconstructed function
#    fig2 = plt.figure()
#    ax2 = fig2.add_subplot(111, projection='3d')
#    
#    surf2 = ax2.plot_surface(ll, mm, Id, rstride=1, cstride=1, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False,alpha=0.5)
#    ax2.set_zlim(Id.min(), Id.max())
#    
#    ax2.zaxis.set_major_locator(LinearLocator(10))
#    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##%%    
#    #Plot reconstructed function
#    fig3 = plt.figure()
#    ax3 = fig3.add_subplot(111, projection='3d')
#    
#    surf3 = ax3.plot_surface(ll, mm, Id2, rstride=1, cstride=1, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False,alpha=0.5)
#    ax3.set_zlim(Id2.min(), Id2.max())
#    
#    ax3.zaxis.set_major_locator(LinearLocator(10))
#    ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##%%    
#    fig4 = plt.figure()
#    ax4 = fig4.add_subplot(111, projection='3d')
#    
#    surf4 = ax4.plot_surface(ll, mm, Idflat.reshape(Np,Np), rstride=1, cstride=1, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False,alpha=0.5)
#    ax4.set_zlim(Idflat.min(), Idflat.max())
#    
#    ax4.zaxis.set_major_locator(LinearLocator(10))
#    ax4.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#    
#    #fig4.colorbar(surf4, shrink=0.5, aspect=5)
#%%
# import numpy as np
#
# Nch = 12
# x = np.linspace(0.7623,1.2376,Nch)
# y0 = x**(-0.7)
# yf = 0.01*y0
# N = 10
# p = np.zeros(4)
# y = y0.copy()
# while all(y > yf):
#     p += 0.1*np.polyfit(x,y,3)
#     y -= 0.1*np.polyval(p,x)
#
# p2 = np.polyfit(x,y0,3)
#
# print p2
# print p
#
# #%%
#
# import numpy as np
# Nch = 12
#
# nu = np.linspace(1.029e9,1.671e9,Nch)
# nu0 = 1.35e9
# x = nu/nu0
# y0 = x**(-0.7)
#
# Fpol = np.array([1.23,1.16,1.09,1.02,0.94,0.86,0.78,0.70,0.63,0.56,0.48,0.41])
# JN = np.array([0.252,0.221,0.198,0.176,0.156,0.139,0.124,0.111,0.0999,0.0899,0.08,0.071])
# plt.plot(x,y0,label=r'$\alpha = -0.7$')
# plt.plot(x,Fpol,'ko', label=r'Fpol=ID/JN')
# plt.plot(x,JN,'xr',label=r'JN')
# plt.legend()
#
# #%%
#
# import numpy as np
# Nch = 12
#
# nu = np.linspace(1.029e9,1.671e9,Nch)
# nu0 = 1.35e9
# x = nu/nu0
# y0 = x**(-0.7)
#
# Fpol = np.array([1.23,1.16,1.09,1.02,0.94,0.86,0.78,0.70,0.63,0.56,0.48,0.41])
# JN = np.array([0.252,0.221,0.198,0.176,0.156,0.139,0.124,0.111,0.0999,0.0899,0.08,0.071])
# #Fpol2 = Fpol*np.sqrt(JN)/JN
# plt.plot(x,y0,label=r'$\alpha = -0.7$')
# plt.plot(x,Fpol,'ko', label=r'$FpolTrue=\frac{I_D}{\sqrt{JN}}$')
# #plt.plot(x,Fpol2,'bo', label=r'Fpol=ID/JN')
# plt.plot(x,JN,'xr',label=r'$JN$')
# plt.xlabel(r'$\frac{\nu}{\nu_0}$', fontsize=24)
# plt.ylabel(r'$I$', fontsize=24)
# plt.legend()
#
# #%%
# Nch = 5
# PSF = np.zeros([Nch,1,3,3])
# for i in xrange(Nch):
#     PSF[i,0,:,:] = np.random.randn(3,3)
#     print PSF[i,0].max()
#
# PSFmax = np.amax(PSF.reshape(Nch,1,3**2),axis=2,keepdims=True).reshape(Nch,1,1,1)


#%%