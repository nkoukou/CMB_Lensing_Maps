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

def mexHat(R, cb): # !!! currently only depends on scale
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
            newmask = filterMask(MAP, scale, cbbd=10)
        newmap = (newmap, newmask)
    return newmap

def filterMask(MAP, scale, cbbd=10):
    '''
    Filters mask according to Planck 2013 XXIII section 4.5.
    # !!! Separate GalPlane from PointSources first in Nside=2048, then 
    degrade to 1024 then follow procedure. m2 is not working
    '''
    R = np.radians(scale)
        
    # Check res and degrades to immediate lower resolution
    MAP.set_res(MAP.res/2)
    cb, lon =  MAP.cb, MAP.lon
    
    aux = np.copy(MAP.mask)
    
            
    # Isolate Galactic plane
    bdu, bdd = np.radians(90-cbbd), np.radians(90+cbbd)
    
    cbu, lonu = cb[cb<bdu], lon[cb<bdu]
    cbd, lond = cb[cb>bdd], lon[cb>bdd]
    
    pixsu = hp.ang2pix(MAP.res, cbu, lonu)
    pixsd = hp.ang2pix(MAP.res, cbd, lond)
    
    aux[pixsu] = aux[pixsd] = 1
    
    # Find and extend boundaries by twice the aperture
    m1 = np.copy(aux)
    
    for pix in np.where(m1==0)[0]:
        if 1 not in m1[hp.get_all_neighbours(MAP.res, pix)]:
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

"""


class FilterMap(object):
    '''
    Contains the filters.
    '''
    def __init__(self, res, lmax, scale, coef):
        '''
        Parameters scale and coef vary the filters.
        '''
        self.res = res
        self.lmax = lmax
        self.R = np.radians(scale)
        self.a = coef
        
        cb, lon = hp.pix2ang(res, np.arange(hp.nside2npix(res)))
        self.cb = cb
        self.lon = lon
        W = mexHat(self.R, self.cb)
        self.wlm = hp.map2alm(W, lmax=self.lmax, mmax=0)
        self.ellFac = np.sqrt(4*np.pi/(2.*np.arange(self.lmax+1)+1))
        self.fl = self.ellFac * np.conj(self.wlm)
    
    def set_res(self, res):
        self.res = res
        cb, lon = hp.pix2ang(res, np.arange(hp.nside2npix(res)))
        self.cb = cb
        self.lon = lon
        W = mexHat(self.R, self.cb)
        self.wlm = hp.map2alm(W, lmax=self.lmax, mmax=0)
        self.fl = self.ellFac * np.conj(self.wlm)
    
    def set_R(self, R):
        self.R = np.radians(R)
        W = mexHat(self.R, self.cb)
        self.wlm = hp.map2alm(W, lmax=self.lmax, mmax=0)
        self.fl = self.ellFac * np.conj(self.wlm)
    
    def set_lmax(self, lmax):
        self.lmax = lmax
        self.ellFac = np.sqrt(4*np.pi/(2.*np.arange(self.lmax+1)+1))
        W = mexHat(self.R, self.cb)
        self.wlm = hp.map2alm(W, lmax=lmax, mmax=0)
        self.fl = self.ellFac * np.conj(self.wlm)
    
    def set_a(self, a):
        self.a = a

    def filterMap(self, MAP, Map):
        '''
        Applies SMHW filter on map. (Vielva, 2010)
        '''   
        if MAP.map[0]==Map[0]:
            mlm = MAP.alm
        else:
            mlm = hp.map2alm(Map, lmax=self.lmax)
        
        convAlm = hp.almxfl(alm=mlm, fl=self.fl)
        newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
        return newmap

    def filterMask(self, MAP, cbbd=10, write=False):
        '''
        Filters mask according to Planck 2013 XXIII section 4.5.
        # !!! Separate GalPlane from PointSources first in Nside=2048, then 
        degrade to 1024 then follow procedure. m2 is not working
        '''
        # Check res = 2048 and degrade to 1024
        #if MAP.res!=2048: raise ValueError('Resolution must be 2048')
        MAP.set_res(MAP.res/2)
        cb, lon =  MAP.findPixs(mode='a')
        aux = np.copy(MAP.mask)
        
                
        # Isolate Galactic plane
        bdu, bdd = np.radians(90-cbbd), np.radians(90+cbbd)
        
        cbu, lonu = cb[cb<bdu], lon[cb<bdu]
        cbd, lond = cb[cb>bdd], lon[cb>bdd]
        
        pixsu = hp.ang2pix(MAP.res, cbu, lonu)
        pixsd = hp.ang2pix(MAP.res, cbd, lond)
        
        aux[pixsu] = aux[pixsd] = 1
        
        # Find and extend boundaries by twice the aperture
        m1 = np.copy(aux)
        
        for pix in np.where(m1==0)[0]:
            if 1 not in m1[hp.get_all_neighbours(MAP.res, pix)]: continue
            vec = hp.pix2vec(MAP.res, pix)
            pixs = hp.query_disc(MAP.res, vec, 2*self.R)
            m1[pixs] = 0
        
        # Convolve with SMHW
        '''
        m2 = np.copy(aux)
        mlm = hp.map2alm(m2, lmax=self.lmax)
        
        W = mexHat(self.R, CB)
        wlm = hp.map2alm(W, lmax=self.lmax, mmax=0)
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
        
        if write:
            np.save(DIR+STR(MAP.res)+'e_fmask', MAP.mask * m)
        
        return MAP.mask * m
        
    def plotFilt(self, MAP, mask=False):
        '''
        Plots given filtered map. Parameters include:
        - mask: if True, mask from filtMask() method is applied.
        - Mbd: Only pixels of mask value >Mbd are considered before filtering
        - Nbd: Only pixels of mask value >Nbd are considered after filtering
        '''
        Map = self.filterMap(MAP, MAP.map)
        if mask:
            mask = self.filterMask(MAP)
            Map[mask==0] = hp.UNSEEN
            Map = hp.ma(Map)
        hp.mollview(Map, title='Filtered CMB T (scale={0:.1f} deg)'.format(
        np.rad2deg(self.R)), cbar=True, unit=r'$K$')
"""

























