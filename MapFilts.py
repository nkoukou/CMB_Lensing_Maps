'''
Includes the filters used to analyse map statistics.

As of commit 14, the module applies on Lensing Maps (MAP has methods kmap, 
klm instead of map, alm as well as maskGal which affects the isolation of the 
Galactic plane step in _filterMask; refer to earlier verisons of the module for
application on temperature maps - with the TempColdSpot.py module).
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
import os.path

# Global constants and functions
STR4 = lambda res: str(int(res)).zfill(4)
STR2 = lambda res: str(int(res)).zfill(2)

###

# Map filtering and plotting functions
def filterMap(MAP, scale, a, is_sim, Gauss=False, mask=True, lmax=None):
    '''
    Filters map at given scale, considering ell components up to given lmax. If 
    lmax = None, the default lmax of the MAP is used.
    
    If mask=True, returns filtered mask as well.
    If is_sim=True, filters simulation MAP.sim instead of the real data.
    
    !!! same mask used regardless the filter (param a or gaussian/top hat)
    '''
    if lmax is not None:
        temp = MAP.lmax
        MAP.lmax = lmax
    
    if is_sim:    
        Map = MAP.sim
        mlm = MAP.slm
    else:    
        Map = MAP.map
        mlm = MAP.alm
    
    if Gauss:
        newmap = hp.smoothing(Map, fwhm=np.radians(scale), pol=False, 
                              verbose=False)
    else:
        fl = np.load(MAP.core+'_wlm'+STR4(60*scale)+STR2(a)+'.npy')
        convAlm = hp.almxfl(alm=mlm, fl=fl)
        newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    
    if mask:
        fmask = MAP.core+'_maskFilt'+STR4(60*scale)+'.npy'
        if os.path.isfile(fmask):
            newmask = np.load(fmask)
        else:
            print('MASK WAS NOT FOUND')
            newmask = _filterMask(MAP, scale, W, ellFac)
        newmap = (newmap, newmask)
    if lmax is not None:
        MAP.lmax = temp
    return newmap

def plotMap(filtmap, filtmask, scale, a):
    '''
    Plots given map with given mask. Both must have already been filtered at 
    given scale and a.
    '''
    res = hp.npix2nside(filtmap.size)
    
    Map = np.copy(filtmap)
    Map[filtmask==0.] = hp.UNSEEN
    Map = hp.ma(Map)
    
    ttl = (r'Filtered map at $(R, \alpha) = $' 
           r'({0}, {1}) and $N_{{side}} = 2048$'.format(scale, a))
    fmt = '%07.3e'
    unt = r'$\kappa$'
    cmap = ListedColormap(np.loadtxt('Figures/cmb_cmap.txt')/255.)
    cmap.set_under('w')
    cmap.set_bad('gray')
    hp.mollview(Map, title=ttl, format=fmt, cmap=cmap, cbar=True, unit=unt)

# Filters
def A(R, a):
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
    W = A(R,a) * J * ( a - 1./RR * yy ) * np.exp( -1./(a*RR) * yy )
    
    return W

def topHat(R, cb):
    '''!!!
    Computes spherical top-hat function for scale R and array of colatitudes cb.
    '''
    A = 3/(4*np.pi*R**3)

# Helper function for mask filtering
def _filterMask(MAP, scale, W, ellFac, m1Fac=2, m2bd=0.1):
    '''
    Filters mask at given scale according to Planck 2013 XXIII (section 4.5).
    Similar to Zhang, Huterer 2010 in the convolution step.
    
    Degrading is not performed because it has little effect on efficiency and 
    results, and thus takes W and ellFac from filterMap() to improve efficiency.
    '''
    R = np.radians(scale)
        
    # Degrade to immediately lower resolution - No degrading currently
    #MAP.set_res(MAP.res/2)
    
    # Isolate Galactic plane
    aux = np.copy(MAP.maskGal)
    
    # Find and extend boundaries by twice the aperture
    m1 = np.copy(aux)
    
    for pix in np.where(m1==0)[0]:
        if 1 not in m1[hp.get_all_neighbours(MAP.res, pix)]:
            continue
        vec = hp.pix2vec(MAP.res, pix)
        pixs = hp.query_disc(MAP.res, vec, m1Fac*R)
        m1[pixs] = 0
    
    # Convolve with SMHW
    m2 = np.copy(aux)
    mlm = hp.map2alm(m2, lmax=MAP.lmax)
    
    wwlm = hp.map2alm(W*W, lmax=MAP.lmax, mmax=0)
    fl = ellFac * np.conj(wwlm)
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    
    m2 = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    m2[m2<m2bd] = 0
    m2[m2>=m2bd] = 1
    
    # Multiply all masks together
    m = m2 * m1

    #res = 2*MAP.res # No degrading currently
    #m = hp.ud_grade(m, res, power=0) # !!!Can use better upgrade method
    #MAP.set_res(res)
    
    newmask = MAP.mask * m
    
    np.save(MAP.core+'_maskFilt'+STR4(60*scale), MAP.mask * m)
    return newmask

# Exporting function for wlm's

def _exportWlm(MAP, scales=np.linspace(0.5, 15, 30), 
              alphas=np.linspace(1, 10, 10)):
    ellFac = np.sqrt(4*np.pi/(2.*np.arange(MAP.lmax+1)+1))
    for s in scales:
        R = np.radians(s)
        for a in alphas:
            print('R, a = ', s, ', ', a)
            W = mexHat(R, a, MAP.cb)
            wlm = hp.map2alm(W, lmax=MAP.lmax, mmax=0)
            fl = ellFac * np.conj(wlm)
            
            np.save(MAP.core+'_wlm'+STR4(60*s)+STR2(a), fl)

def _plotWprof():
    thetas = np.linspace(0,np.pi/2, 10000, endpoint=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    W = mexHat(np.radians(10), 2, thetas)
    ax.plot(thetas*180/np.pi, W, label=r'$\alpha$ = {0}'.format(102))
    W = mexHat(np.radians(10), 6, thetas)
    ax.plot(thetas*180/np.pi, W, label=r'$\alpha$ = {0}'.format(106))
    W = mexHat(np.radians(15), 2, thetas)
    ax.plot(thetas*180/np.pi, W, label=r'$\alpha$ = {0}'.format(152))
    W = mexHat(np.radians(15), 6, thetas)
    ax.plot(thetas*180/np.pi, W, label=r'$\alpha$ = {0}'.format(156))
    
    ax.set_xlabel(r'$\theta$ (deg)', fontsize=14)
    ax.set_ylabel(r'$W(\theta; R, \alpha)$', fontsize=14)
    ax.set_xlim([0,90])
    ax.set_ylim([-0.2,1])
    ax.legend(loc='upper right', prop={'size':14})

















