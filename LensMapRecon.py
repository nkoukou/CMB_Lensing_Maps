"""
Reconstruction of CMB Lensing Map from Planck 2015 data release.
"""
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

# Global constants and functions
DIR = '/media/nikos/00A076B9A076B52E/Users/nkoukou/Desktop/UBC/'
RAWSPEC = np.loadtxt(DIR+'data/aux/nlkk.dat')
CMB_CMAP = np.loadtxt(DIR+'data/aux/cmb_cmap.txt')/255.

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
    def __init__(self, phi, conserv=False, res=2048):
        '''        
        Reads the lensing map of given resolution from expected directory. Then,
        all necessary secondary data are also loaded (e.g. mask). If res=None, 
        the original .fits file is read with resolution 2048. 
        '''
        self.phi = phi
        self.conserv = conserv
                
        if res==None:
            self.mask = hp.read_map(DIR+'data/aux/mask.fits', verbose=False)
            self.malm = hp.map2alm(self.mask)
            print('MASK')

            self.maskGal = hp.read_map(DIR+'data/aux/maskGal.fits', field=3, 
                                       verbose=False)
            self.malmGal = hp.map2alm(self.maskGal)
            print('MASKGAL')
            
            self.res = hp.npix2nside(self.mask.size)
            self.lmax = LMAX(self.res) #!!!
            
            self.klm = hp.read_alm(DIR+'data/aux/dat_klm.fits')
            self.kmap = hp.alm2map(self.klm, self.res, verbose=False)
            print('KMAP')
            
            self.nlkk = RAWSPEC[:self.lmax+1,1]
            self.clkk = RAWSPEC[:self.lmax+1,2] - RAWSPEC[:self.lmax+1,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.flm = hp.almxfl(self.klm, fl)
            self.fmap = hp.alm2map(self.flm, self.res, verbose=False)
            self.clff = fl**2 * self.clkk
            print('FMAP')
        
        elif res in NSIDES:
            direc = DIR+'data/maps/n'+STR4(res)
            
            self.res = res
            self.lmax = LMAX(res)
            
            self.cb = np.load(direc+'a_cb.npy')
            self.lon = np.load(direc+'a_lon.npy')
            
            #!!! Downgrading affects noise correlations and thus nlkk and clkk
            self.nlkk = RAWSPEC[:self.lmax+1,1]
            self.clkk = RAWSPEC[:self.lmax+1,2] - RAWSPEC[:self.lmax+1,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.clff = fl**2 * self.clkk
            
            if not phi:
                self.map = np.load(direc+'_kmap.npy')
                self.alm = np.load(direc+'_klm.npy')
            else:
                self.map = np.load(direc+'_fmap.npy')
                self.alm = np.load(direc+'_flm.npy')
            
            if self.conserv:
                fl = np.concatenate(( np.zeros(40), np.ones(361), np.zeros(1648) )) #!!!
                hp.almxfl(self.alm, fl, inplace=True)
                self.map = hp.alm2map(self.alm, self.res, pol=False, verbose=False)
            
            self.sim = None
            
            self.mask = np.load(direc+'a_mask.npy')
            self.malm = np.load(direc+'a_malm.npy')
            
            self.maskGal = np.load(direc+'a_maskGal.npy')
            self.malmGal = np.load(direc+'a_malmGal.npy')
            
        else:
            raise ValueError('Resolution (Nside) must be a power of 2')
    
    def set_res(self, res):
        '''
        Resets object with new resolution.
        '''
        self.__init__(res=int(res))
        
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
        
        print('Half low')
        
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
        
        self.cb, self.lon = hp.pix2ang(res, np.arange(self.kmap.size))
        
        print('Full low')
    
    def _write(self, res):
        '''
        Produces output data files of map and mask data for given resolution.
        '''
        if res!=2048: self._lowRes(res)
        
        np.save(DIR+'data/maps/n'+STR4(res)+'_kmap', self.kmap)
        np.save(DIR+'data/maps/n'+STR4(res)+'_klm', self.klm)
        
        np.save(DIR+'data/maps/n'+STR4(res)+'_fmap', self.fmap)
        np.save(DIR+'data/maps/n'+STR4(res)+'_flm', self.flm)
        
        print('Half write')
        
        np.save(DIR+'data/maps/n'+STR4(res)+'a_mask', self.mask)
        np.save(DIR+'data/maps/n'+STR4(res)+'a_malm', self.malm)
        
        np.save(DIR+'data/maps/n'+STR4(res)+'a_maskGal', self.maskGal)
        np.save(DIR+'data/maps/n'+STR4(res)+'a_malmGal', self.malmGal)
        
        np.save(DIR+'data/maps/n'+STR4(res)+'a_cb', self.cb)
        np.save(DIR+'data/maps/n'+STR4(res)+'a_lon', self.lon)

        print('Full write')
    
    def _writeAll(self):
        '''
        Produces output data files of map and mask data for all resolutions.
        '''
        for res in NSIDES[:-1]:
            print(res)
            self._write(res)
            self.set_res(None)
    
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
        
        cmap = ListedColormap(CMB_CMAP)
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
    
    def plotNoisySpec(self, sims):
        '''
        Compares power spectrum of given simulations with data.
        '''
        print('START')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        clm = hp.anafast(self.map, pol=False)
        cl = [clm]
        ax.plot(clm, 'k.', label='Data')
        for s in sims:
            print(s)
            self.loadSim(s)
            print('CL')
            cls = hp.anafast(self.sim, pol=False)
            cl.append(cls)
            ax.plot(cls, '.', label='Sim '+str(s))
        ax.set_title('Kappa Power Spectrum')
        ax.legend(loc='upper right', prop={'size':14})
        return cl
    
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
        
        if self.conserv:
            fl = np.concatenate(( np.zeros(40), np.ones(361), np.zeros(1648) ))
            hp.almxfl(self.slm, fl, inplace=True)
            self.sim = hp.alm2map(self.slm, self.res, pol=False, verbose=False)
            
        if plot:
            Map = np.copy(self.sim)
            Map[self.mask==0.] = hp.UNSEEN
            Map = hp.ma(Map)
            if self.phi: title = r'Simulated lensing potential $\phi$'
            else: title = r'Simulated lensing convergence $\kappa$'
            hp.mollview(Map, coord='G', title=title, cbar=True, 
                        unit=r'dimensionless')









