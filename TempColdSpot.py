"""
Detects Cold Spot on Temperature Map of Planck 2015

x1. downgrade map
x2. take power spectrum
x3. generate simulations (correct for ell)
4. filter map with mexican hat
5. Add noise, convolutions, rotations
6. compare features of data with simulaitons
7. apply mask
"""

import numpy as npfilter
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt

DirMap = 'COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits'
DirPow = 'COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt'

class TempMap(object):
    """
    Represents the CMB Lensing Map of the Planck 2015 data release (ref. Planck
    2015 results XV. Gravitational lensing).
    """
    def __init__(self, res=None):
        '''
        Checks for Planck data at the expected directory ('/home2/nkoukou/data')
        and then imports:
        
        - mask: lensing potential map mask #!!! reference
        - klm: spherical harmonic coefficients of lensing convergence kappa #!!! units?
        - rawSpec: approximate noise and signal+noise power spectra of kappa #!!! units?
        '''
        self.name = 'CMBT_'
        try:
            if res is None:
                direc = DirMap
                self.map = hp.read_map(direc, field=0)
                self.mask = hp.read_map(direc, field=1)
                self.hdu = ap.io.fits.open(direc)
            else:
                self.res = res
                self.map = hp.read_map(self.name+'map_n'+str(res)+'.fits')
                self.mask = hp.read_map(self.name+'mask_n'+str(res)+'.fits')
        except:
            print('Files are not in expected directory ' + 
                  'or do not have expected filename')
        self.NSIDE = 2048 # !!! refer to header/readme
        lm = hp.Alm.getlm(self.NSIDE)
        self.ELL = lm[0]
        self.EM = lm[1]
        
        self.cl = None
        self.alm = None
    
    def printHeaders(self, h=None, hdulist=[0, 1, 2]):
        '''
        Prints all headers of the map.
        '''
        if h is None:
            for i in hdulist:
                print(self.hdu[i].header)
        elif h in hdulist:
            print(self.hdu[h])
        else:
            print('Invalid header')
    
    def write(self, Map=False, Mask=False, Cl=False, Alm=False, fname=None):
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
        
        Mask may be optimally downgraded in other ways. !!!
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
        self.cl = hp.anafast(self.map)
        self.alm = hp.map2alm(self.map)
        
        if write:
            res = hp.get_nside(self.map)
            self.write(Cl=True, fname=self.name+'cl_n'+str(res)+'.fits')
            self.write(Alm=True, fname=self.name+'alm_n'+str(res)+'.fits')
    
    def plotMap(self, mask=False):
        '''
        Plots map.
        
        Mask is applied when value < 1. !!!
        '''
        Map = np.copy(self.map)
        if mask:
            Map[self.mask<1.] = hp.UNSEEN
            Map = hp.ma(Map)
        hp.mollview(Map, title='CMB Temperature', cbar=True, unit=r'$K$')

    def genSim(self, lmax=None, plot=False):
        '''
        Generates a simulation from the theoretical temperature power spectrum
        (no noise). If lmax is set to None, all components of the map are used 
        for the simulation.
        '''
        self.calcSpectrum(write=False)
        if lmax is None: lmax = self.cl.size
        
        spec = np.loadtxt(DirPow, comments='#', delimiter=None, usecols=(0,1))
        ell, cls = spec[:lmax, 0], spec[:lmax, 1]
        
        cls = 2*np.pi/(ell*(ell+1)) * cls #!!! correction
        
        sim = hp.synfast(cls, self.res, alm=False, pol=True, pixwin=False,
                         fwhm=0., sigma=None, verbose=True)
        sim = 1e-6 * sim #Convert uK to K
        
        if plot:
            hp.mollview(sim, title='Simulated CMB T', cbar=True,
                        unit=r'$K$ (*factor)')
        return sim















