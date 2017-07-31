'''
Includes the filters used to analyse map statistics.
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt

# Global constants and functions
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


class FilterMap(object):
    '''
    Contains the filters.
    '''
    def __init__(self, lmax, scale, coef):
        '''
        Parameters scale and coef vary the filters.
        '''
        self.lmax = lmax
        self.R = np.radians(scale)
        self.a = coef

    def filterMap(self, MAP, sim=False):
        '''
        Applies SMHW filter on map. (Vielva, 2010)
        '''   
        if sim:
            Map = MAP.genSim(lmax=self.lmax)
            mlm = hp.map2alm(Map)
        else:
            Map = MAP.map
            mlm = MAP.alm
        cb, lon = MAP.allPixs(mode='a')
        
        ellFac = np.sqrt(4*np.pi/(2.*np.arange(self.lmax+1)+1))
        W = mexHat(self.R, cb)
        wlm = hp.map2alm(W, lmax=self.lmax, mmax=0)
        fl = ellFac * np.conj(wlm)
        
        convAlm = hp.almxfl(alm=mlm, fl=fl)
        newmap = hp.alm2map(convAlm, nside=MAP.res, pol=False, verbose=False)
        return newmap

    def filterMask(self, MAP, cbbd=10):
        '''
        Filters mask according to Planck 2013 XXIII section 4.5.
        # !!! Separate GalPlane from PointSources first in Nside=2048, then 
        degrade to 1024 then follow procedure. m2 is not working
        '''
        # Check res = 2048 and degrade to 1024
        if MAP.res!=2048: raise ValueError('Resolution must be 2048')
        MAP.set_res(MAP.res/2)
        cb, lon =  MAP.allPixs(mode='a')
        aux = np.copy(MAP.mask)
        
                
        # Isolate Galactic plane
        bdu, bdd = np.radians(90-cbbd), np.radians(90+cbbd)
        
        cbu, lonu = cb[cb<bdu], lon[cb<bdu]
        cbd, lond = cb[cb>bdd], lon[cb>bdd]
        
        pixsu = hp.ang2pix(MAP.res, cbu, lonu)
        pixsd = hp.ang2pix(MAP.res, cbd, lond)
        
        aux[pixsu] = aux[pixsd] = 1
        
        return aux
        '''
        # Find and extend boundaries by twice the aperture
        m1 = np.copy(aux)
        m1[m1<0.5] = 0
        m1[m1>0.5] = 1
        
        bds = []
        for pix in np.where(m1==0):
            if 1 in m1[hp.get_all_neighbours(self.res, pix)]:
                bds.append(pix)
        
        vecs = hp.pix2vec(self.res, np.array(bds))
        vecs = np.vstack((vecs[0], vecs[1], vecs[2])).T
        for vec in vecs:
            pixs = hp.query_disc(self.res, vec, 2*R)
            m1[pixs] = 0
        
        # Convolve with SMHW
        m2 = np.copy(aux)
        mlm = hp.map2alm(m2)
        
        W = mexHat(R, cb)
        wlm = hp.map2alm(W)[:lmax+1]
        
        ellFac = np.sqrt(4*np.pi/(2.*np.arange(lmax+1)+1))
        fl = ellFac*np.conj(wlm)
        convAlm = hp.almxfl(alm=mlm, fl=fl)
        
        m2 = hp.alm2map(convAlm, nside=self.res, pol=False, verbose=False)
        m2[m2<0.1] = 0
        m2[m2>0.1] = 1
        
        self.mask = m*m1
        '''
        
    def plotFilt(self, Map=None, mask=False, Mbd=0.9, Nbd=0.9):
        '''
        Plots given filtered map. Parameters include:
        - mask: if True, mask from filtMask() method is applied.
        - Mbd: Only pixels of mask value >Mbd are considered before filtering
        - Nbd: Only pixels of mask value >Nbd are considered after filtering
        '''
        Map = self.filtMap(Map)
        if mask:
            mask = self.filtMask(Mbd)
            #Map[mask<Nbd] = hp.UNSEEN
            Map[self.mask==0] = hp.UNSEEN
            Map = hp.ma(Map)
        hp.mollview(Map, title='Filtered CMB T (scale={0})'.format(self.R), 
                    cbar=True, unit=r'$K$')






























