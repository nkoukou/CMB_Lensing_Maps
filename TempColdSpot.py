"""
Loads Temperature Map of Planck 2015 data release 
(ref. Planck 2015 results XVI)..
"""

import numpy as np
import scipy.signal as ss
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
import os.path

# Global constants and functions
DIRMAP = 'CMBT_Maps/COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits'
DIRPOW = 'CMBT_Maps/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt'

COORDCS = (210, -57) #lon, lat in degrees in Galactic coordinates
BEAM = 5./60 * np.pi/180 #radians

NSIDES = [2**x for x in range(4, 12)]
FWHM = {};
for nside in NSIDES: FWHM[nside] = BEAM * (2048/nside) #radians

LMAX = lambda res: 3*res - 1
STR = lambda res: str(res).zfill(4)

###

class TempMap(object):
    '''
    Represents the CMB Temperature Map of the Planck 2015 data release.
    '''
    def __init__(self, res=None):
        '''
        Read the temperature map of given resolution from expected directory. 
        If res=None, the original fits file is read with resolution 2048. Then, 
        all necessary secondary data are also loaded (e.g. mask).
        '''
        self._overwrite = 1
        self.dir = 'CMBT_Maps/n'
        self.hdu = ap.io.fits.open(DIRMAP)
        
        if res is None:
            self.map = hp.read_map(DIRMAP, field=0)
            self.mask = hp.read_map(DIRMAP, field=1)
            self.res = hp.get_nside(self.map)
            
        elif res in NSIDES:
            self.map=hp.read_map(self.dir+STR(res)+'_map.fits', verbose=False)
            print('MAP')
            self.alm = hp.read_alm(self.dir+STR(res)+'_alm.fits')
            print('ALM')
            self.cl = hp.read_cl(self.dir+STR(res)+'_cl.fits')
            print('CL')
            
            self.mask=hp.read_map(self.dir+STR(res)+'_mask.fits', verbose=False)
            print('MASK')
            self.malm = hp.read_alm(self.dir+STR(res)+'_malm.fits')
            print('MALM')
            self.mcl = hp.read_cl(self.dir+STR(res)+'_mcl.fits')
            print('MCL')
            
            self.res = res
            
        else:
            raise ValueError('Resolution (Nside) must be a power of 2')
        
        self.lmax = LMAX(self.res)
        lm = hp.Alm.getlm(self.lmax)
        print('LM')
        self.ELL = lm[0]
        self.EM = lm[1]
        
        if not os.path.isfile(self.dir+str(NSIDES[-1])+'_map.fits'):
            res = NSIDES[-1]
            self.cl, self.alm = hp.anafast(self.map, alm=True, pol=False)
            fname = self.dir+STR(res)+'_map.fits'
            hp.write_map(fname, self.map, nest=False)
            fname = self.dir+STR(res)+'_alm.fits'
            hp.write_alm(fname, self.alm)
            fname = self.dir+STR(res)+'_cl.fits'
            hp.write_cl(fname, self.cl)
            
            self.mcl, self.malm = hp.anafast(self.mask, alm=True, pol=False)
            fname = self.dir+STR(res)+'_mask.fits'
            hp.write_map(fname, self.mask, nest=False)
            fname = self.dir+STR(res)+'_malm.fits'
            hp.write_alm(fname, self.malm)
            fname = self.dir+STR(res)+'_mcl.fits'
            hp.write_cl(fname, self.mcl)
    
    # Setters and exporting methods
    def set_res(self, res):
        '''
        Resets object with new resolution.
        '''
        self.__init__(int(res))
        
    def printHeaders(self, h=None):
        '''
        Prints all headers of the map fits file.
        '''
        if h is None:
            for i in range(len(self.hdu)):
                print(self.hdu[i].header)
        elif h in range(len(self.hdu)):
            print(self.hdu[h].header)
        else:
            raise IndexError('Invalid header index')
    
    def _calcSpec(self):
        '''
        Calculates spectra and alm coefficients from temperature map and mask 
        (including noise).
        '''
        if not self._overwrite: raise Exception('Not allowed to overwite')
        
        self.cl = hp.anafast(self.map, alm=False, pol=False)
        self.mcl = hp.anafast(self.mask, alm=False, pol=False)
    
    def _lowRes(self, res, heal=False, bd=0.9):
        '''
        Downgrades resolution of map (ref. Planck 2015 results XVI. Section 2).
        '''
        if not self._overwrite: raise Exception('Not allowed to overwite')
        
        if heal:
            self.map = hp.ud_grade(self.map, res, power=0)
            self.mask = hp.ud_grade(self.mask, res, power=0)
        
        if not heal:
            lmax = LMAX(res)
            
            beam0 = hp.gauss_beam(FWHM[self.res], lmax)
            pixw0 = hp.pixwin(self.res)[:lmax+1]
            beam = hp.gauss_beam(FWHM[res], lmax)
            pixw = hp.pixwin(res)[:lmax+1]
            fl = (beam*pixw)/(beam0*pixw0)
            
            self.alm = hp.almxfl(self.alm, fl)
            self.malm = hp.almxfl(self.malm, fl)
            
            lowmap = hp.alm2map(self.alm, res, verbose=False)
            lowmask = hp.alm2map(self.malm, res, verbose=False)
            lowmask[lowmask<bd] = 0
            lowmask[lowmask>=bd] = 1
            
            self.map = lowmap
            self.mask = lowmask
    
    def _write(self, res, Map=False, Mask=False):
        '''
        Produces output data files of Map or Mask data or both for given 
        resolution.
        '''
        if not self._overwrite: raise Exception('Not allowed to overwite')            
        
        self._lowRes(res)
        self._calcSpec()
        
        if Map:
            fname = self.dir+STR(res)+'_map.fits'
            hp.write_map(fname, self.map, nest=False)
            fname = self.dir+STR(res)+'_alm.fits'
            hp.write_alm(fname, self.alm)
            fname = self.dir+STR(res)+'_cl.fits'
            hp.write_cl(fname, self.cl)
        if Mask:
            fname = self.dir+STR(res)+'_mask.fits'
            hp.write_map(fname, self.mask, nest=False)
            fname = self.dir+STR(res)+'_malm.fits'
            hp.write_alm(fname, self.malm)
            fname = self.dir+STR(res)+'_mcl.fits'
            hp.write_cl(fname, self.mcl)
    
    def _writeAll(self):
        if not self._overwrite: raise Exception('Not allowed to overwite') 
        for res in NSIDES[:-1]:
            self.set_res(NSIDES[-1])
            self._write(res, Map=True, Mask=True)
    
    # Map methods
    def findPixs(self, idxs=None, mode='p'):
        '''
        Returns all indices of pixels of the map. Parameter mode can be:
          - 'p': returns indices of pixels
          - 'v': returns Cartesian unit vectors instead
          - 'a': returns colatitude-longitude instead 
        '''
        if idxs is None: idxs = np.arange(self.map.size)
        
        if mode=='p':
            pixs = idxs
        elif mode=='v':
            pixs = hp.pix2vec(self.res, idxs)
            pixs = np.vstack((pixs[0], pixs[1], pixs[2])).T
        elif mode=='a':
            pixs = hp.pix2ang(self.res, idxs)
        else:
            raise ValueError("Mode can be 'p', 'v' or 'a'")
        
        return pixs
    
    def plotMap(self, mask=False):
        '''
        Plots map including mask if True.
        '''
        Map = np.copy(self.map)
        if mask:
            Map[self.mask==0.] = hp.UNSEEN
            Map = hp.ma(Map)
        hp.mollview(Map, coord='G', title='CMB Temperature', cbar=True, 
                    unit=r'$K$')

    def genSim(self, lmax=None, plot=False, mask=False):
        '''
        Generates a simulation from the theoretical temperature power spectrum
        (no additional noise realisation is added apart from instrumental). If 
        lmax is set to None, all spherical components are used for the 
        simulation.
        '''
        if lmax is None: lmax = self.cl.size-1
        
        spec = np.loadtxt(DIRPOW, comments='#', delimiter=None, usecols=(0,1))
        ell, cls = spec[:lmax+1, 0], spec[:lmax+1, 1]
        
        cls = 2*np.pi/(ell*(ell+1)) * cls #!!! Correction (must be correct)
        
        sim = hp.synfast(cls, self.res, alm=False, pol=False, pixwin=False,
                         fwhm=FWHM[self.res], sigma=None, verbose=False)

        #Convert uK to K
        sim = 1e-6 * sim
        
        if plot:
            Map = np.copy(sim)
            if mask:
                Map[self.mask==0.] = hp.UNSEEN
                Map = hp.ma(Map)
            hp.mollview(Map, coord='G', title='Simulated CMB T', cbar=True,
                        unit=r'$K$')
        return sim
















