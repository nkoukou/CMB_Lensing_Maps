'''
Includes the filters used to analyse map statistics.
'''
import os.path
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

# Global constants and functions
DIR = '/media/nikos/00A076B9A076B52E/Users/nkoukou/Desktop/UBC/'
dirfig = lambda ttl: DIR+'drafts/b_draft/figs/'+ttl+'.pdf'
CMB_CMAP = np.loadtxt(DIR+'data/aux/cmb_cmap.txt')/255.

NSIDES = [2**x for x in range(4, 12)]
BEAM = 5./60 * np.pi/180 #radians (ref. 
                         #Planck 2015 results XV. Gravitational lensing)
FWHM = {}; # (ref. as above)
for nside in NSIDES: FWHM[nside] = BEAM * (2048/nside) #radians

M1FAC = 2.00
M2BDS = 0.10
M2BDG = 0.95

STR2 = lambda res: str(int(res)).zfill(2)
STR4 = lambda res: str(int(res)).zfill(4)

def fmt(x):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

###

# Map filtering and plotting functions
def filterMap(MAP, scale, a, mask=True):
    '''
    Filters map at given scale in arcmins. If mask=True, returns filtered mask 
    as well. The map is current MAP loaded simulation.
    '''
    R = np.radians(scale/60)
    
    if not a:
        w = gauss(R, MAP.cb)
        #fl = A*np.exp(-0.5 * ell * (ell + 1) * sigma ** 2)
        fmask = MAP.dir+'a_maskFiltG'+STR4(scale)+'.npy'
    else:        
        w = mexHat(R, a, MAP.cb)
        fmask = MAP.dir+'a_maskFiltS'+STR4(scale)+'.npy'
    wlm = hp.map2alm(w, lmax=MAP.lmax, mmax=0)
    ellFac = np.sqrt(4*np.pi/(2.*np.arange(MAP.lmax+1)+1))
    fl = ellFac * np.conj(wlm)
    convAlm = hp.almxfl(alm=MAP.slm, fl=fl)
    newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    
    if mask:
        if os.path.isfile(fmask):
            newmask = np.load(fmask)
        else:
            print('MASK WAS NOT FOUND')
            newmask = _filterMask(MAP, scale, a)
        newmap = (newmap, newmask)
    return newmap

def plotMap(filtmap, filtmask, phi, ttl=None):
    '''
    Plots given map with given mask. Both must have already been filtered at 
    given scale and a.
    '''
    res = hp.npix2nside(filtmap.size)
    
    Map = np.copy(filtmap)
    Map[filtmask==0.] = hp.UNSEEN
    Map = hp.ma(Map)
    
    if phi is None: unt = r'$T (K)$'
    elif phi: unt = r'$\phi$'
    else: unt = r'$\kappa$'
    
    #ttl = (r'Filtered map at $(R, \alpha) = $' 
    #       r'({0}, {1}) and $N_{{side}} = 'r'{2}'r'$'.format(scale, a, res))
    ticks = np.linspace(0.99*Map.min(), 0.99*Map.max(), 2)
    
    cmap = ListedColormap(CMB_CMAP)
    cmap.set_under('w')
    cmap.set_bad('gray')
    
    hp.mollview(Map, title = '', cmap=cmap, cbar=False)
    
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    cbar = fig.colorbar(image, ax=ax, orientation='horizontal', pad=0.04,
           fraction=0.05, aspect=26, ticks=[Map.min(),Map.max()])
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([fmt(Map.min()), fmt(Map.max())])
    cbar.ax.set_xlabel(unt, fontsize=16, labelpad=-18)
    cbar.ax.tick_params(labelsize=16, pad=10, size=0)
    if ttl is not None: plt.savefig(dirfig(ttl), bbox_inches='tight')

# Gauss
def gauss(R,cb):
    '''
    Computes Gaussian function for FWHM R and array of colatitudes cb.
    '''
    sigma = R/( 2.*np.sqrt(2.*np.log(2.)) )
    G = np.exp( - cb*cb /(2*sigma*sigma) )
    A = np.sqrt( 1./( 2*np.pi*np.trapz(G*G*np.sin(cb), cb) ) )
    #A = 1./( 2*np.pi*np.trapz(G*np.sin(MAP.cb), MAP.cb) )
    return A*G

# SMHWs
def Amp(R, a):
    '''
    Normalisation amplitude so that mexHat squared integrates to unity.
    '''
    RR = R*R
    A = 1./np.sqrt(2*np.pi*RR*(a**3/8. + a**4/32. * RR + a**5/128. * RR*RR))
    return A
    
def mexHat(R, a, cb):
    '''
    Computes SMHW function for scale R, parameter a and array of colatitudes cb.
    '''
    # Transformation of variable
    y = 2*np.tan(cb/2.)
    
    # Squares
    yy = y*y
    RR = R*R
    
    # Stereographic projection
    J = (1. + yy/4.)*(1. + yy/4.)
    
    # Wavelet function
    W = Amp(R,a) * J * ( a - 1./RR * yy ) * np.exp( -1./(a*RR) * yy )
    
    return W

# Helper functions for masking
def _filterMask(MAP, scale, a):
    '''
    Filters mask at given scale according to Planck 2013 XXIII (section 4.5).
    Similar to Zhang, Huterer 2010 in the convolution step.
    
    Degrading is not performed because it has little effect on efficiency and 
    results. !!!
    '''
    R = np.radians(scale/60)
    
    # Isolate Galactic plane for res=2048
    res = 2048; bd =0.9 # Magic constants (bd referes to last step - downgrade)
    aux = np.load(DIR+'data/maps/n'+STR4(res)+'a_maskGal.npy')
    
    # Find and extend boundaries by a factor M1FAC of the aperture
    m1 = DIR+'data/aux/n'+STR4(res)+'_R'+STR4(scale)+'_m1.npy'
    if os.path.isfile(m1):
        m1 = np.load(m1)
    else:
        fmask = m1
        m1 = np.copy(aux)
        for pix in np.where(m1==0)[0]:
            if 1 not in m1[hp.get_all_neighbours(res, pix)]:
                continue
            vec = hp.pix2vec(res, pix)
            pixs = hp.query_disc(res, vec, M1FAC*R)
            m1[pixs] = 0
        np.save(fmask, m1)
    
    # Convolve with filter
    mlm = np.load(DIR+'data/maps/n'+STR4(res)+'a_malmGal.npy')
    if not a:
        sigma = R / (2.*np.sqrt(2.*np.log(2.)))
        ell = np.arange(MAP.lmax + 1.)
        gauss = np.sin(MAP.cb)*np.exp( - MAP.cb*MAP.cb /(2*sigma*sigma) )
        A = 1./( 2*np.pi*np.trapz(gauss, MAP.cb) )
        fl = np.exp(-0.5 * ell * (ell + 1) * sigma ** 2)
        bound = M2BDG
        fmask = MAP.dir+'a_maskFiltG'+STR4(scale)+'.npy'
    else:     
        W = mexHat(R, a, MAP.cb)
        wlm = hp.map2alm(W*W, lmax=MAP.lmax, mmax=0)
        ellFac = np.sqrt(4*np.pi/(2.*np.arange(MAP.lmax+1)+1))
        fl = ellFac * np.conj(wlm)
        bound = M2BDS
        fmask = MAP.dir+'a_maskFiltS'+STR4(scale)+'.npy'
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    m2 = hp.alm2map(convAlm, nside=res, pol=False, verbose=False)
    m2[m2<bound] = 0
    m2[m2>=bound] = 1
    
    # Multiply the masks together
    m = m2 * m1
    mlm = hp.map2alm(m, lmax=MAP.lmax)

    # Upgrading to 2048
    #beam0 = hp.gauss_beam(FWHM[res], MAP.lmax)
    #pixw0 = hp.pixwin(res)[:MAP.lmax+1]
    #beam = hp.gauss_beam(FWHM[2*res], MAP.lmax)
    #pixw = hp.pixwin(2*res)[:MAP.lmax+1]
    #fl = (beam*pixw)/(beam0*pixw0)
    #
    #hp.almxfl(mlm, fl, inplace=True)
    #m = hp.alm2map(mlm, nside=2*res, pol=False, verbose=False)
    
    # Downgrading to MAP.res
    beam0 = hp.gauss_beam(FWHM[res], MAP.lmax)
    pixw0 = hp.pixwin(res)[:MAP.lmax+1]
    beam = hp.gauss_beam(FWHM[MAP.res], MAP.lmax)
    pixw = hp.pixwin(MAP.res)[:MAP.lmax+1]
    fl = (beam*pixw)/(beam0*pixw0)
    
    hp.almxfl(mlm, fl, inplace=True)
    m = hp.alm2map(mlm, nside=MAP.res, pol=False, verbose=False)
    m[m<bd] = 0
    m[m>=bd] = 1
    
    newmask = MAP.mask * m
    np.save(fmask, newmask)
    return newmask

def _filterAllMasks(lmap):
    '''
    Writes all masks for the filtered maps
    '''
    scales = np.array([10, 25, 50, 100, 750, 1000])
    scales = np.concatenate((scales, np.linspace(30, 900, 30)))
    for scale in scales:
        for a in [0,1]:
            print('SCALE, A : ', scale, a)
            nmp, nmk= filterMap(lmap, scale, a, mask=True)
    

# Plot function for wavelet profiles
def profileFilts():
    '''
    Plots the profiles of SMHW and Gaussian filters.
    '''
    thetas = np.linspace(0,np.pi/2, 10000, endpoint=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for r in [1, 5, 10, 15, 20]:
        R = np.radians(r)
        W = mexHat(R, 2, thetas)/(2*Amp(R,2))
        ax.plot(thetas*180/np.pi, W, label=r'$R$ = {0}'.format(r))
    ax.set_xlabel(r'$\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$W(\theta; R, \alpha=2)$', fontsize=12)
    ax.set_xlim([0, 80])
    ax.set_ylim([-0.2,1])
    ax.legend(loc='upper right', prop={'size':12})
    plt.savefig(dirfig('smhwR_prof'), bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    R = np.radians(5)
    for a in [1, 2, 4, 7, 10]:
        W = mexHat(R, a, thetas)/(a*Amp(R,a))
        ax.plot(thetas*180/np.pi, W, label=r'$\alpha$ = {0}'.format(a))
    ax.set_xlabel(r'$\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$W(\theta; R=5^\circ, \alpha)$', fontsize=12)
    ax.set_xlim([0, 45])
    ax.set_ylim([-0.2,1])
    ax.legend(loc='upper right', prop={'size':12})
    plt.savefig(dirfig('smhwA_prof'), bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for r in [1, 5, 10, 15, 20]:
        R = np.radians(r)
        sigma = R/( 2.*np.sqrt(2.*np.log(2.)) )
        G = np.exp( - thetas*thetas /(2*sigma*sigma) )
        ax.plot(thetas*180/np.pi, G, label=r'$R$ = {0}'.format(r))
    ax.set_xlabel(r'$\theta$ (deg)', fontsize=12)
    ax.set_ylabel(r'$G(\theta; R)$', fontsize=12)
    ax.set_xlim([0, 35])
    ax.set_ylim([-0.05,1])
    ax.legend(loc='upper right', prop={'size':12})
    plt.savefig(dirfig('gauss_prof'), bbox_inches='tight')

# Test wavelets
def waveletTest():
    '''
    Tests wavelet properties of Gaussian and SMHW filters.
    '''
    thetas = np.linspace(0,np.pi/2, 10000, endpoint=False)
    normG = np.zeros(5)
    compG = np.ones(5)
    normW = np.zeros((5,5))
    compW = np.ones((5,5))
    scales = [1, 5, 10, 15, 20]
    alphas = [1, 2, 4, 7, 10]
    for i in range(len(scales)):
        R = np.radians(scales[i])
        G = gauss(R, thetas)
        compG[i] = 2*np.pi*np.trapz(G*np.sin(thetas), thetas)
        normG[i] = 2*np.pi*np.trapz(G*G*np.sin(thetas), thetas)
        for j in range(len(alphas)):
            W = mexHat(R, alphas[j], thetas)
            compW[i,j] = 2*np.pi*np.trapz(W*np.sin(thetas), thetas)
            normW[i,j] = 2*np.pi*np.trapz(W*W*np.sin(thetas), thetas)
    return normG, compG, normW, compW
        
        














