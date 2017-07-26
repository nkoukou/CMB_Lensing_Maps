"""
Detects Cold Spot on Temperature Map of Planck 2015

!!!
- Apply better mask while downgrading and plotting
"""

import numpy as np
import scipy.signal as ss
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt

DIRMAP = 'COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits'
DIRPOW = 'COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt'
BEAM = 5./60 * np.pi/180

class TempMap(object):
    """
    Represents the CMB Lensing Map of the Planck 2015 data release (ref. Planck
    2015 results XV. Gravitational lensing).
    """
    def __init__(self, res=None):
        '''
        Tries to read the temperature map of given resolution from expected
        directory. If res=None, the resilution is 2048. Then, all necessary raw
        data are loaded (map values, mask and alm components along with less
        significant data).
        '''
        self.name = 'CMBT_'
        self.hdu = ap.io.fits.open(DIRMAP)
        
        try:
            if res is None:
                self.map = hp.read_map(DIRMAP, field=0)
                self.mask = hp.read_map(DIRMAP, field=1)
                self.res = hp.get_nside(self.map)
            else:
                self.res = res
                self.map = hp.read_map(self.name+'map_n'+str(res)+'.fits', 
                                       verbose=False)
                self.mask = hp.read_map(self.name+'mask_n'+str(res)+'.fits',
                                        verbose=False)
        except:
            raise FileNotFoundError('No such file or directory')
        
        lm = hp.Alm.getlm(3*self.res-1)
        self.ELL = lm[0]
        self.EM = lm[1]
        
        self.calcSpectrum(write=False)
    
    def set_res(self, res):
        if res==self.res: return
        try:
            self.res = res
            self.map = hp.read_map(self.name+'map_n'+str(res)+'.fits', 
                                   verbose=False)
            self.mask = hp.read_map(self.name+'mask_n'+str(res)+'.fits',
                                    verbose=False)
        except:
            raise FileNotFoundError('No such file or directory')
            
        lm = hp.Alm.getlm(3*self.res-1)
        self.ELL = lm[0]
        self.EM = lm[1]
        
        self.calcSpectrum(write=False)
        
    def printHeaders(self, h=None, hdulist=[0, 1, 2]):
        '''
        Prints all headers of the map.
        '''
        if h is None:
            for i in hdulist:
                print(self.hdu[i].header)
        elif h in hdulist:
            print(self.hdu[h].header)
        else:
            print('Invalid header')
    
    def write(self, Map=False, Mask=False, Cl=False, Alm=False, fname=None):
        '''
        Produces output data files of data types set to True.
        '''
        if Map:
            if fname is None: fname = self.name+'map.fits'
            hp.write_map(fname, self.map, nest=False)
        if Mask:
            print('mask')
            if fname is None: fname = self.name+'mask.fits'
            hp.write_map(fname, self.mask, nest=False)
        if Cl:
            if fname is None: fname = self.name+'cl.fits'
            hp.write_cl(fname, self.cl)
        if Alm:
            if fname is None: fname = self.name+'alm.fits'
            hp.write_alm(fname, self.alm)
    
    def lowRes(self, res, write=False):
        '''
        Downgrades resolution of map.
        
        Mask may be optimally downgraded in other ways.
        '''
        self.map = hp.ud_grade(self.map, res, power=0)
        self.mask = hp.ud_grade(self.mask, res, power=0)
        
        if write:
            self.write(Map=True, fname=self.name+'map_n'+str(res)+'.fits')
            self.write(Mask=True, fname=self.name+'mask_n'+str(res)+'.fits')
    
    def calcSpectrum(self, write=False):
        '''
        Calculates spectrum from temperature map (including noise).
        '''
        self.cl, self.alm = hp.anafast(self.map, lmax=None, mmax=None, 
                                       alm=True, pol=False)
        
        if write:
            self.write(Cl=True, fname=self.name+'cl_n'+str(self.res)+'.fits')
            self.write(Alm=True, fname=self.name+'alm_n'+str(self.res)+'.fits')
    
    def plotMap(self, mask=False):
        '''
        Plots map.
        
        Mask is applied when value < 1.
        '''
        Map = np.copy(self.map)
        if mask:
            Map[self.mask<1.] = hp.UNSEEN
            Map = hp.ma(Map)
        hp.mollview(Map, coord='G', title='CMB Temperature', cbar=True, 
                    unit=r'$K$')

    def genSim(self, lmax=None, plot=False, mask=False):
        '''
        Generates a simulation from the theoretical temperature power spectrum
        (no noise). If lmax is set to None, all components of the map are used 
        for the simulation.
        '''
        if lmax is None: lmax = self.cl.size-1
        
        spec = np.loadtxt(DIRPOW, comments='#', delimiter=None, usecols=(0,1))
        ell, cls = spec[:lmax+1, 0], spec[:lmax+1, 1]
        
        cls = 2*np.pi/(ell*(ell+1)) * cls #!!! Correction (must be correct)
        
        sim = hp.synfast(cls, self.res, alm=False, pol=False, pixwin=False,
                         fwhm=BEAM, sigma=None, verbose=False)

        #Convert uK to K
        sim = 1e-6 * sim
        
        if plot:
            Map = np.copy(sim)
            if mask:
                Map[self.mask<1.] = hp.UNSEEN
                Map = hp.ma(Map)
            hp.mollview(Map, title='Simulated CMB T', cbar=True,
                        unit=r'$K$')
        return sim
















