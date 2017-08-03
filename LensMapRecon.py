"""
Reconstruction of CMB Lensing Map from Planck 2015 data release.

!!! FIX:
- simulations
- test stats
"""
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt

# Global constants and functions
DIR_MASKGAL = 'CMBL_Maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits'

NSIDES = [2**x for x in range(4, 12)]

BEAM = 5./60 * np.pi/180 #radians
FWHM = {};
for nside in NSIDES: FWHM[nside] = BEAM * (2048/nside) #radians

LMAX = lambda res: res
STR = lambda res: str(int(res)).zfill(4)

###

class LensingMap(object):
    """
    Represents the CMB Lensing Map of the Planck 2015 data release (ref. Planck
    2015 results XV. Gravitational lensing).
    """
    def __init__(self, res=2048):
        '''        
        Reads the lensing map of given resolution from expected directory. Then,
        all necessary secondary data are also loaded (e.g. mask). If res=None, 
        the original .fits file is read with resolution 2048. 
        '''
        self.dir = 'CMBL_Maps/'
        self.core = self.dir + 'data/n' + STR(res)
        rawSpec = np.loadtxt(self.dir+'nlkk.dat')
        
        if res==None:
            self.mask = hp.read_map(self.dir+'mask.fits', verbose=False)
            self.malm = hp.map2alm(self.mask)
            print('MASK')
            self.maskGal = hp.read_map(DIR_MASKGAL, field=3, verbose=False)
            self.malmGal = hp.map2alm(self.maskGal)
            print('MASKGAL')
            
            self.res = hp.npix2nside(self.mask.size)
            self.lmax = LMAX(self.res)
            
            self.klm = hp.read_alm(self.dir+'dat_klm.fits')
            self.kmap = hp.alm2map(self.klm, self.res, verbose=False)
            print('KMAP')
            
            self.nlkk = rawSpec[:self.lmax+1,1]
            self.clkk = rawSpec[:self.lmax+1,2] - rawSpec[:self.lmax+1,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.flm = hp.almxfl(self.klm, fl)
            self.fmap = hp.alm2map(self.flm, self.res, verbose=False)
            self.clff = fl**2 * self.clkk
            print('FMAP')
            
        elif res in NSIDES:
            self.res = res
            self.lmax = LMAX(res)
            
            self.cb = np.load(self.core+'_cb.npy')
            self.lon = np.load(self.core+'_lon.npy')
            
            #!!! Downgrading affects noise correlations and thus nlkk and clkk
            self.nlkk = rawSpec[:self.lmax+1,1]
            self.clkk = rawSpec[:self.lmax+1,2] - rawSpec[:self.lmax+1,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.clff = fl**2 * self.clkk
            print('KMAP')
            
            self.kmap = hp.read_map(self.core+'_kmap.fits', verbose=False)
            self.klm = hp.read_alm(self.core+'_klm.fits')
            print('KMAP -> FMAP')
            
            self.fmap = hp.read_map(self.core+'_fmap.fits', verbose=False)
            self.flm = hp.read_alm(self.core+'_flm.fits')
            print('FMAP -> MASK')
            
            self.mask = hp.read_map(self.core+'_mask.fits', verbose=False)
            #self.malm = hp.read_alm(self.core+'_malm.fits')
            print('MASK -> MASKGAL')
            
            self.maskGal = hp.read_map(self.core+'_maskGal.fits', verbose=False)
            #self.malmGal = hp.read_alm(self.core+'_malmGal.fits')
            
            self.sim = None
            
        else:
            raise ValueError('Resolution (Nside) must be a power of 2')
    
    def set_res(self, res):
        '''
        Resets object with new resolution.
        '''
        self.__init__(int(res))
        
    def _lowRes(self, res, bd=0.9):
        '''
        Downgrades resolution of map (ref. Planck 2015 results XVI. Section 2).
        '''
        lmax = LMAX(res)
        beam0 = hp.gauss_beam(FWHM[self.res], lmax)
        pixw0 = hp.pixwin(self.res)[:lmax+1]
        beam = hp.gauss_beam(FWHM[res], lmax)
        pixw = hp.pixwin(res)[:lmax+1]
        fl = (beam*pixw)/(beam0*pixw0)
        
        self.klm = hp.almxfl(self.klm, fl)
        self.flm = hp.almxfl(self.flm, fl)
        self.malm = hp.almxfl(self.malm, fl)
        self.malmGal = hp.almxfl(self.malmGal, fl)
            
        lowkmap = hp.alm2map(self.klm, res, verbose=False)
        lowfmap = hp.alm2map(self.flm, res, verbose=False)
        lowmask = hp.alm2map(self.malm, res, verbose=False)
        lowmaskGal = hp.alm2map(self.malmGal, res, verbose=False)
        lowmask[lowmask<bd] = 0
        lowmask[lowmask>=bd] = 1
        lowmaskGal[lowmaskGal<bd] = 0
        lowmaskGal[lowmaskGal>=bd] = 1
        
        self.kmap = lowkmap
        self.fmap = lowfmap
        self.mask = lowmask
        self.maskGal = lowmaskGal
    
    def _write(self, res):
        '''
        Produces output data files of map and mask data for given resolution.
        '''
        if res!=2048: self._lowRes(res)
        
        hp.write_map(self.core+'_kmap.fits', self.kmap, nest=False)
        hp.write_alm(self.core+'_klm.fits', self.klm)
        
        hp.write_map(self.core+'_fmap.fits', self.fmap, nest=False)
        hp.write_alm(self.core+'_flm.fits', self.flm)
        
        hp.write_map(self.core+'_mask.fits', self.mask, nest=False)
        hp.write_alm(self.core+'_malm.fits', self.malm)
        
        hp.write_map(self.core+'_maskGal.fits', self.maskGal, nest=False)
        hp.write_alm(self.core+'_malmGal.fits', self.malmGal)
    
    def _writeAll(self):
        '''
        Produces output data files of map and mask data for all resolutions.
        '''
        for res in NSIDES[:-1]:
            print(res)
            self.set_res(NSIDES[-1])
            self._write(res)
    
    def plotMap(self, phi=False, mask=False):
        '''
        Plots map of phi or kappa and includes mask if mask=True.
        '''
        if phi:
            Map = np.copy(self.fmap)
            title = r'Lensing potential $\phi$'
        else:
            Map = np.copy(self.kmap)
            title = r'Lensing convergence $\kappa$'
        if mask:
            Map[self.mask==0.] = hp.UNSEEN
            Map = hp.ma(Map)
        hp.mollview(Map, coord='G', title=title+' at res = '+str(self.res), 
                    cbar=True, unit=r'dimensionless')
    
    def plotSpec(self):
        '''
        Plots figure 6 of Planck 2015 results XV.
        '''
        start = 8
        ell = np.arange(self.lmax+1)
        y = 2./np.pi * 1e7 * self.clkk
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(ell[start:], y[start:])
        ax.set_xlabel(r'$L$')  
        ax.set_ylabel(r'$\frac{[L(L+1)]^2}{2\pi}C_L^{\phi\phi}\ [\times 10^7]$')
    
    def genSim(self, lmax=None, plot=False, mask=False):
        '''
        Generates a simulation !!!
        '''
        sim = None
        
        if plot:
            Map = np.copy(sim)
            if mask:
                Map[self.mask==0.] = hp.UNSEEN
                Map = hp.ma(Map)
            hp.mollview(Map, coord='G', title='Simulated phi or kapa', 
                        cbar=True, unit=r'dimensionless')
        self.sim = sim









