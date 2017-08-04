'''
Includes the filters used to analyse map statistics.

As of this version, the module applies on Lensing Maps (MAP has methods kmap, 
klm instead of map, alm as well as maskGal which affects the isolation of the 
Galactic plane step in _filterMask; refer to earlier verisons of the module for
application on temperature maps - with the TempColdSpot.py module).
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
import os.path

# Global constants and functions
STR = lambda res: str(int(res)).zfill(4)

###

def mexHat(R, cb):
    '''
    Computes SMHW function for scale R and array of co-latitudes cb.
    '''
    # Transformation of variable - ignores conventional factor of 2
    y = np.tan(cb/2.)
    
    # Squares
    yy = y*y
    RR = R*R
    
    # Normalisation coefficient for square of wavelet
    A = 1./np.sqrt(2*np.pi*RR*(1. + RR/2. + RR*RR/4.))
    
    # Wavelet function
    W = A * (1. + yy)*(1. + yy) * (2. - 4./RR * yy) * np.exp(-2./RR * yy)
    
    return W

def filterMap(MAP, lmax, scale, mask=False, sim=False):
    '''
    Filters map at given scale, considering ell components up to given lmax.
    
    If mask=True, returns filtered mask as well.
    If sim=True, filters simulation MAP.sim instead of the real data.
    !!! perhaps save all sims created or create many sims over night - check 
        space
    '''
    MAP.lmax = lmax
    R = np.radians(scale)
    if sim:
        Map = MAP.sim
        mlm = hp.map2alm(Map, lmax=MAP.lmax)
    else:
        Map = MAP.kmap
        mlm = MAP.klm
    
    W = mexHat(R, MAP.cb)
    wlm = hp.map2alm(W, lmax=MAP.lmax, mmax=0)
    
    ellFac = np.sqrt(4*np.pi/(2.*np.arange(MAP.lmax+1)+1))
    fl = ellFac * np.conj(wlm)
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    
    newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    
    if mask:
        fmask = MAP.core+'_maskFilt'+STR(60*scale)+'.npy'
        if os.path.isfile(fmask):
            newmask = np.load(fmask)
        else:
            newmask = _filterMask(MAP, scale, W, ellFac)
        newmap = (newmap, newmask)
    return newmap

def _filterMask(MAP, scale, W, ellFac):
    '''
    Filters mask at given scale according to Planck 2013 XXIII (section 4.5).
    Similar to Zhang, Huterer 2010 in the convolution step.
    
    Degrading is not performed because it has little effect on efficiency and 
    results, and thus takes W and ellFac from filterMap() to improve efficiency.
    '''
    R = np.radians(scale)
        
    # Degrade to immediately lower resolution !!! not done currently
    #MAP.set_res(MAP.res/2)
    
    # Isolate Galactic plane
    aux = np.copy(MAP.maskGal)
    
    # Find and extend boundaries by twice the aperture
    m1 = np.copy(aux)
    
    for pix in np.where(m1==0)[0]:
        if 1 not in m1[hp.get_all_neighbours(MAP.res, pix)]:
            continue
        vec = hp.pix2vec(MAP.res, pix)
        pixs = hp.query_disc(MAP.res, vec, 2*R)
        m1[pixs] = 0
    
    # Convolve with SMHW
    m2 = np.copy(aux)
    mlm = hp.map2alm(m2, lmax=MAP.lmax)
    
    wwlm = hp.map2alm(W*W, lmax=MAP.lmax, mmax=0)
    fl = ellFac * np.conj(wwlm)
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    
    m2 = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    m2[m2<0.1] = 0
    m2[m2>=0.1] = 1
    
    # Multiply all masks together
    m = m2 * m1

    #res = 2*MAP.res #!!! No degrading
    #m = hp.ud_grade(m, res, power=0) #!!! better upgrade method
    #MAP.set_res(res)
    
    newmask = MAP.mask * m
    
    np.save(MAP.core+'_maskFilt'+STR(60*scale), MAP.mask * m)
    return newmask

def plotMap(fmap, fmask, R):
    '''
    Plots given map with given mask. Both must have already been filtered at 
    scale R.
    '''
    res = hp.npix2nside(fmap.size)
    
    Map = np.copy(fmap)
    Map[fmask==0.] = hp.UNSEEN
    Map = hp.ma(Map)
    
    hp.mollview(Map, title = r'Nside = {0}, scale = {1} deg'.format(res, R))
    plt.show()
























