"""
Reconstruction of CMB Lensing Map from Planck 2015 data release.

As of commit 27 underscored methods are depricated.
"""
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

# Global constants and functions
NSIDES = [2**x for x in range(4, 12)]
NSIMS = 99

BEAM = 5./60 * np.pi/180 #radians
FWHM = {};
for nside in NSIDES: FWHM[nside] = BEAM * (2048/nside) #radians

LMAX = lambda res: res
STR4 = lambda res: str(int(res)).zfill(4)

###

class LensingMap(object):
    """
    Represents the CMB Lensing Map of the Planck 2015 data release (ref. Planck
    2015 results XV. Gravitational lensing).
    """
    def __init__(self, phi, res=2048):
        '''        
        Reads the lensing map of given resolution from expected directory. Then,
        all necessary secondary data are also loaded (e.g. mask). If res=None, 
        the original .fits file is read with resolution 2048. 
        '''
        self.dir = 'CMBL_Maps/'
        self.core = self.dir + 'data/n' + STR4(res)
        self.dirSim = self.dir + 'sims/obs_klms/'
        rawSpec = np.loadtxt(self.dir+'nlkk.dat')
        
        self.phi = phi
        if res==None:
            self.mask = hp.read_map(self.dir+'.none/mask.fits', verbose=False)
            self.malm = hp.map2alm(self.mask)
            print('MASK')
            DIR_MASKGAL='CMBL_Maps/.none/HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
            self.maskGal = hp.read_map(DIR_MASKGAL, field=3, verbose=False)
            self.malmGal = hp.map2alm(self.maskGal)
            print('MASKGAL')
            
            self.res = hp.npix2nside(self.mask.size)
            self.lmax = LMAX(self.res)
            
            self.klm = hp.read_alm(self.dir+'.none/dat_klm.fits')
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
            
            if not phi:
                self.map = hp.read_map(self.core+'_kmap.fits', verbose=False)
                self.alm = hp.read_alm(self.core+'_klm.fits')
            else:
                self.map = hp.read_map(self.core+'_fmap.fits', verbose=False)
                self.alm = hp.read_alm(self.core+'_flm.fits')
            self.sim = None
            
            self.mask = hp.read_map(self.core+'_mask.fits', verbose=False)
            #self.malm = hp.read_alm(self.core+'_malm.fits')
            
            #self.maskGal = hp.read_map(self.core+'_maskGal.fits',verbose=False)
            #self.malmGal = hp.read_alm(self.core+'_malmGal.fits')
            
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
    
    def plotMap(self, mask=True):
        '''
        Plots map of phi or kappa and includes mask if mask=True.
        '''
        Map = np.copy(self.map)
        if self.phi:
            ttl = r'Lensing potential $\phi$'
            fmt = '%07.3e'
            unt = r'$\phi$'
        else:
            ttl = r'Lensing convergence $\kappa$'
            fmt = '%.3f'
            unt = r'$\kappa$'
        if mask:
            Map[self.mask==0.] = hp.UNSEEN
            Map = hp.ma(Map)
        
        cmap = ListedColormap(np.loadtxt('Figures/cmb_cmap.txt')/255.)
        cmap.set_under('w')
        cmap.set_bad('gray')
        hp.mollview(Map, title=ttl+' at $N_{side} = 2048$', 
                    format=fmt, cmap=cmap, cbar=True, unit=unt)
    
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
    
    def loadSim(self, n, plot=False):
        '''
        Loads the n-th simulation in directory self.dirSim.
        
        !!!only in res=2048, how do sim_lm give rise to 0 values in the area 
        covered by mask??
        '''
        if self.res!=2048: raise ValueError('Only res=2048 considered')
        
        if n==99:
            self.slm = self.alm
            self.sim = self.map
            return
        
        slm = hp.read_alm(self.dirSim+'sim_'+STR4(n)+'_klm.fits')
        if self.phi:
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.slm = hp.almxfl(slm, fl)
            self.sim = np.load(self.dirSim+'sim_'+STR4(n)+'_fmap.npy')
        else:
            self.slm = slm
            self.sim = np.load(self.dirSim+'sim_'+STR4(n)+'_kmap.npy')
            
        if plot:
            Map = np.copy(self.sim)
            Map[self.mask==0.] = hp.UNSEEN
            Map = hp.ma(Map)
            if self.phi: title = r'Simulated lensing potential $\phi$'
            else: title = r'Simulated lensing convergence $\kappa$'
            hp.mollview(Map, coord='G', title=title, cbar=True, 
                        unit=r'dimensionless')




