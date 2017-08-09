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
import os.path

# Global constants and functions
STR4 = lambda res: str(int(res)).zfill(4)
STR2 = lambda res: str(int(res)).zfill(2)

###

# Map filtering and plotting functions
def filterMap(MAP, scale, a, phi, mask=False, sim=False, lmax=None):
    '''
    Filters map at given scale, considering ell components up to given lmax. If 
    lmax = None, the default lmax of the MAP is used.
    
    If mask=True, returns filtered mask as well.
    If sim=True, filters simulation MAP.sim instead of the real data.
    
    !!! same mask used regardless the value of parameter a
    '''
    R = np.radians(scale)
    if lmax is not None:
        temp = MAP.lmax
        MAP.lmax = lmax
    
    if sim and phi:    
        Map = MAP.fsim
        mlm = hp.map2alm(Map, lmax=MAP.lmax)
    elif sim and not phi:    
        Map = MAP.ksim
        mlm = hp.map2alm(Map, lmax=MAP.lmax)
    elif not sim and phi:    
        Map = MAP.fmap
        mlm = MAP.flm
    elif not sim and not phi:    
        Map = MAP.kmap
        mlm = MAP.klm
    else:
        raise ValueError('Checl phi and sim arguments')
    #print("Filt")
    W = mexHat(R, a, MAP.cb)
    wlm = hp.map2alm(W, lmax=MAP.lmax, mmax=0)
    
    ellFac = np.sqrt(4*np.pi/(2.*np.arange(MAP.lmax+1)+1))
    fl = ellFac * np.conj(wlm)
    convAlm = hp.almxfl(alm=mlm, fl=fl)
    #print("Conv")
    newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
    
    if mask:
        fmask = MAP.core+'_maskFilt'+STR4(60*scale)+'.npy'
        if os.path.isfile(fmask):
            #print(1)
            newmask = np.load(fmask)
        else:
            print(0)
            newmask = _filterMask(MAP, scale, W, ellFac)
        newmap = (newmap, newmask)
    if lmax is not None:
        MAP.lmax = temp
    return newmap

def plotMap(fmap, fmask, scale, a):
    '''
    Plots given map with given mask. Both must have already been filtered at 
    given scale and a.
    '''
    res = hp.npix2nside(fmap.size)
    
    Map = np.copy(fmap)
    Map[fmask==0.] = hp.UNSEEN
    Map = hp.ma(Map)
    
    hp.mollview(Map, title = r'Nside = {0}, scale = {1} deg, a = {2}'.format(
                res, scale, a))
    #plt.show()

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
    
    return W #/(a*A(R,a))

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
























