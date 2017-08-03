'''
Includes the filters used to analyse map statistics.
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
import os.path

# Global constants and functions
STR = lambda res: str(res).zfill(4)

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
    MAP.lmax = lmax
    R = np.radians(scale)
    if sim:
        Map = MAP.sim
        mlm = hp.map2alm(Map, lmax=lmax)
    else:
        Map = MAP.map
        mlm = MAP.alm
    
    cb = MAP.cb
    W = mexHat(R, cb)
    wlm = hp.map2alm(W, lmax=lmax, mmax=0)
    
    ellFac = np.sqrt(4*np.pi/(2.*np.arange(lmax+1)+1))
    fl = ellFac * np.conj(wlm)
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    
    newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    
    if mask:
        fmask = MAP.dir+STR(MAP.res)+'e_fmask'+STR(60*scale)+'.npy'
        if os.path.isfile(fmask):
            newmask = np.load(fmask)
        else:
            newmask = _filterMask(MAP, scale, cbbd=10)
        newmap = (newmap, newmask)
    return newmap

def _filterMask(MAP, scale, cbbd=10):
    '''
    Filters mask according to Planck 2013 XXIII section 4.5.
    # !!! Separate GalPlane from PointSources first in Nside=2048, then 
    degrade to 1024 then follow procedure. m2 is not working
    '''
    R = np.radians(scale)
        
    # Degrade to immediately lower resolution
    MAP.set_res(MAP.res/2)
    cb, lon =  MAP.cb, MAP.lon
    
    # Isolate Galactic plane
    aux = np.copy(MAP.maskGal)
    
    # Find and extend boundaries by twice the aperture
    m1 = np.copy(aux)
    
    for pix in np.where(m1==0)[0]:
        if 1 not in m1[hp.get_all_neighbours(MAP.res, pix)]:
            #perhaps faster to apply to all pixs
            continue
        vec = hp.pix2vec(MAP.res, pix)
        pixs = hp.query_disc(MAP.res, vec, 2*R)
        m1[pixs] = 0
    
    # Convolve with SMHW
    '''
    m2 = np.copy(aux)
    mlm = hp.map2alm(m2, lmax=MAP.lmax)
    
    W = mexHat(R, CB)
    wlm = hp.map2alm(W, lmax=MAP.lmax, mmax=0)
    fl = self.ellFac*np.conj(wlm)
    
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    m2 = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    m2[m2<0.1] = 0
    m2[m2>=0.1] = 1
    '''
    
    m = m1 #*m2

    res = 2*MAP.res
    m = hp.ud_grade(m, res, power=0)
    MAP.set_res(res)
    
    np.save(MAP.dir+STR(MAP.res)+'e_fmask'+STR(60*scale), MAP.mask * m)
    
    return MAP.mask * m

def plotMap(fmap, fmask, R):
    res = hp.npix2nside(fmap.size)
    
    Map = np.copy(fmap)
    Map[fmask==0.] = hp.UNSEEN
    Map = hp.ma(Map)
    
    hp.mollview(Map, title = r'Nside = {0}, scale = {1} deg'.format(res, R))
    plt.show()
























