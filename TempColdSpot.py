"""
Detects Cold Spot on Temperature Map of Planck 2015

- Put all fits files in new Temp directory, if file exists load it, else create all dependencies
- Cleaner code, more independent classes
- Methods should depend on lmax
"""

import numpy as np
import scipy.signal as ss
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt

DIRMAP = 'COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits'
DIRPOW = 'COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt'
BEAM = 5./60 * np.pi/180
FWHM = {16: 640, 32: 320, 64: 160, 128: 80, 256: 40, 512: 20, 1024: 10, 2048: 5}

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
                self.calcSpectrum(mode='r')
            else:
                self.res = res
                self.map = hp.read_map(self.name+'map_n'+str(res)+'.fits', 
                                       verbose=False)
                self.mask = hp.read_map(self.name+'mask_n'+str(res)+'.fits',
                                        verbose=False)
                self.calcSpectrum(mode='c')
        except:
            raise FileNotFoundError('No such file or directory')
        
        lm = hp.Alm.getlm(3*self.res-1)
        self.ELL = lm[0]
        self.EM = lm[1]
    
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
        
        self.calcSpectrum(mode='c')
        
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
            if fname is None: fname = self.name+'mask.fits'
            hp.write_map(fname, self.mask, nest=False)
        if Cl:
            if fname is None: fname = self.name+'cl.fits'
            hp.write_cl(fname, self.cl)
        if Alm:
            if fname is None: fname = self.name+'alm.fits'
            hp.write_alm(fname, self.alm)
    
    def lowRes(self, res, lmax=0, write=False):
        '''
        Downgrades resolution of map.
        
        Mask may be optimally downgraded in other ways.
        '''
        if lmax:
            lmax = 3*res-1
            cf = np.pi/(180*60)
            beam0 = hp.gauss_beam(cf*FWHM[self.res], lmax)
            beam = hp.gauss_beam(cf*FWHM[res], lmax)
            pixw0 = hp.pixwin(self.res)[:lmax+1]
            pixw = hp.pixwin(res)[:lmax+1]
            
            #idxs = []
            #Lmax = hp.Alm.getlmax(self.alm.size)
            #for ell in range(lmax+1):
            #    for em in range(ell+1):
            #        idxs.append(hp.Alm.getidx(Lmax, ell, em))
            #idxs = np.array(idxs)
            #alm = self.alm[idxs]
            msklm = hp.read_alm('CMBT_maskAlm_n2048.fits')
            
            fl = (beam*pixw)/(beam0*pixw0)
            alm = hp.almxfl(self.alm, fl)
            msklm = hp.almxfl(msklm, fl)
            lowmap = hp.alm2map(alm, res)
            lowmask = hp.alm2map(msklm, res)
            lowmask[lowmask<0.9] = 0
            lowmask[lowmask>=0.9] = 1
            self.map = lowmap
            self.mask = lowmask
        
        if not lmax:
            self.map = hp.ud_grade(self.map, res, power=0)
            self.mask = hp.ud_grade(self.mask, res, power=0)
        
        if write:
            self.write(Map=True, fname=self.name+'map_n'+str(res)+'.fits')
            self.write(Mask=True, fname=self.name+'mask_n'+str(res)+'.fits')
        
        print('setting')
        self.set_res(res)
    
    def calcSpectrum(self, mode='c'):
        '''
        Calculates spectrum from temperature map (including noise).
        '''
        if mode is None: return
        if mode=='c':
            self.cl, self.alm = hp.anafast(self.map, lmax=None, mmax=None, 
                                           alm=True, pol=False)
        if mode=='w':
            self.write(Cl=True, fname=self.name+'cl_n'+str(self.res)+'.fits')
            self.write(Alm=True, fname=self.name+'alm_n'+str(self.res)+'.fits')
        if mode=='r':
            self.cl = hp.read_cl(self.name+'cl_n'+str(self.res)+'.fits')
            self.alm = hp.read_alm(self.name+'alm_n'+str(self.res)+'.fits')
    
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
                Map[self.mask==0.] = hp.UNSEEN
                Map = hp.ma(Map)
            hp.mollview(Map, title='Simulated CMB T', cbar=True,
                        unit=r'$K$')
        return sim
















