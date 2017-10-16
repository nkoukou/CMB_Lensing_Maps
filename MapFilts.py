'''
Includes the filters used to analyse map statistics.
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
import os.path

# Global constants and functions
DIR = '/media/nikos/00A076B9A076B52E/Users/nkoukou/Desktop/UBC/'
CMB_CMAP = np.loadtxt(DIR+'data/aux/cmb_cmap.txt')/255.

M1FAC = 2.00
M2BDS = 0.10
M2BDG = 0.95

STR4 = lambda res: str(int(res)).zfill(4)
STR2 = lambda res: str(int(res)).zfill(2)

###

# Map filtering and plotting functions
def filterMap(MAP, scale, a, mask=True):
    '''
    Filters map at given scale in degrees. If mask=True, returns filtered mask 
    as well. The map is current MAP loaded simulation.
    
    !!! same mask used regardless the filter (param a or gaussian/top hat)
    '''
    Map = MAP.sim
    mlm = MAP.slm
    
    if not a:
        sigma = np.radians(scale) / (2.*np.sqrt(2.*np.log(2.)))        
        ell = np.arange(MAP.lmax + 1.)        
        fl = np.exp(-0.5 * ell * (ell + 1) * sigma ** 2)
    else:
        R = np.radians(scale)        
        W = mexHat(R, a, MAP.cb)
        wlm = hp.map2alm(W, lmax=MAP.lmax, mmax=0)
        ellFac = np.sqrt(4*np.pi/(2.*np.arange(MAP.lmax+1)+1))
        fl = ellFac * np.conj(wlm)
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    
    if mask:
        fmask = MAP.dir+'a_maskFilt'+STR4(60*scale)+'.npy'
        if os.path.isfile(fmask):
            newmask = np.load(fmask)
        else:
            print('MASK WAS NOT FOUND')
            newmask = _filterMask(MAP, R, a, W, ellFac, fmask)
        newmap = (newmap, newmask)
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
    cmap = ListedColormap(CMB_CMAP)
    cmap.set_under('w')
    cmap.set_bad('gray')
    hp.mollview(Map, title=ttl, format=fmt, cmap=cmap, cbar=True, unit=unt)

# SMHWs
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

# Helper function for mask filtering
def _filterMask(MAP, R, a, W, ellFac, fmask):
    '''
    Filters mask at given scale according to Planck 2013 XXIII (section 4.5).
    Similar to Zhang, Huterer 2010 in the convolution step.
    
    Degrading is not performed because it has little effect on efficiency and 
    results, and thus takes W and ellFac from filterMap() to improve efficiency.
    '''
    R = np.radians(scale)
    
#    # Check if first auxiliary mask exists
#    m1 = DIR+
#    if os.path.isfile(m1):
#        m1 = np.load(m1)
#    else:
    # Isolate Galactic plane for res=1024
    aux = np.load(DIR+'data/maps/n1024a_maskGal.npy')
    
    # Find and extend boundaries by a factor M1FAC of the aperture
    m1 = np.copy(aux)
    
    for pix in np.where(m1==0)[0]:
        if 1 not in m1[hp.get_all_neighbours(MAP.res, pix)]:
            continue
        vec = hp.pix2vec(MAP.res, pix)
        pixs = hp.query_disc(MAP.res, vec, M1FAC*R)
        m1[pixs] = 0
    
    # Convolve with filter
    m2 = np.copy(aux)
    mlm = hp.map2alm(m2, lmax=MAP.lmax)
    
    wwlm = hp.map2alm(W*W, lmax=MAP.lmax, mmax=0)
    fl = ellFac * np.conj(wwlm)
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    
    m2 = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    
    if not a:
        bd = M2BDG
    else:
        bd = M2BDS
    m2[m2<bd] = 0
    m2[m2>=bd] = 1
    
    
    # Multiply all masks together
    m = m2 * m1

    #!!! Skip upgrading
    
    newmask = MAP.mask * m
    
    np.save(fmask, newmask)
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
    '''
    !!!
    '''
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

















