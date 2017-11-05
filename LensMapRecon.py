"""
Reconstruction of CMB Lensing Map from Planck 2015 data release.
"""
import os.path
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
NSIMS = 100

NOISY = 40 # (ref. Planck 2015 results XV. Gravitational lensing)
BEAM = 5./60 * np.pi/180 #radians (ref. as above)
FWHM = {}; # (ref. as above)
for nside in NSIDES: FWHM[nside] = BEAM * (2048/nside) #radians

LMAX = lambda res: res if res==NSIDES[-1] else 2*res
STR2 = lambda res: str(int(res)).zfill(2)
STR4 = lambda res: str(int(res)).zfill(4)

###

class LensingMap(object):
    """
    Represents the CMB Lensing Map of the Planck 2015 data release (ref. Planck
    2015 results XV. Gravitational lensing).
    """
    def __init__(self, phi, conserv=False, res=2048):
        '''        
        Reads the lensing map of given resolution. Then, all necessary secondary
        data are also loaded (e.g. mask).
        '''
        self.phi = phi
        self.conserv = conserv
        self.dir = DIR+'data/maps/n'+STR4(res)
        
        if res in NSIDES:            
            self.res = res
            self.lmax = LMAX(res)
            
            self.cb = np.load(self.dir+'a_cb.npy')
            self.lon = np.load(self.dir+'a_lon.npy')
            
            #!!! Downgrading affects noise correlations and thus nlkk and clkk
            self.nlkk = RAWSPEC[:self.lmax+1,1]
            self.clkk = RAWSPEC[:self.lmax+1,2] - RAWSPEC[:self.lmax+1,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.clff = fl**2 * self.clkk
            
            if not phi: #!!! too many trailing 0's
                self.map = np.load(self.dir+'_kmap.npy')
                self.alm = np.load(self.dir+'_klm.npy')
            else:
                self.map = np.load(self.dir+'_fmap.npy')
                self.alm = np.load(self.dir+'_flm.npy')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            if self.conserv:
                self.noiseFilt = np.concatenate(( np.zeros(NOISY),
                     np.ones(self.lmax+1 - NOISY), 
                     np.zeros(NSIDES[-1]-self.lmax) ))
                hp.almxfl(self.alm, self.noiseFilt, inplace=True)
                
                cons_phi = '_fmapC.npy' if self.phi else '_kmapC.npy'
                cons_map = self.dir+cons_phi
                if os.path.isfile(cons_map):
                    self.map = np.load(cons_map)
                else:
                    self.map = hp.alm2map(self.alm, self.res, pol=False, 
                                          verbose=False)
                    np.save(cons_map, self.map)
            
            self.sim = None
            
            self.mask = np.load(self.dir+'a_mask.npy')
            self.malm = np.load(self.dir+'a_malm.npy')
            
            self.maskGal = np.load(self.dir+'a_maskGal.npy')
            self.malmGal = np.load(self.dir+'a_malmGal.npy')
            
        else:
            raise ValueError('Resolution (Nside) must be a power of 2')
    
    def __repr__(self):
        return 'LensingMap(phi={0}, conserv={1}, res={2})'.format(
               self.phi, self.conserv, self.res)
    
    def set_phi(self, phi):
        '''
        Resets object with new phi parameter (either True or False).
        '''
        if (self.phi==phi or phi is None): return
        self.__init__(phi=phi, conserv=self.conserv, res=self.res)
    
    def set_conserv(self, conserv):
        '''
        Resets object with new conserv parameter (either True or False).
        '''
        if (self.conserv==conserv or conserv is None): return
        self.__init__(phi=self.phi, conserv=conserv, res=self.res)
    
    def set_res(self, res):
        '''
        Resets object with new resolution.
        '''
        if (self.res==res or res is None): return
        self.__init__(phi=self.phi, conserv=self.conserv, res=int(res))
    
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
        hp.mollview(Map, title=ttl+' at $N_{{side}} = {0}$'.format(self.res), 
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
        if self.res not in [256, 2048]:
            raise ValueError('Only res=256 or 2048 considered')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        clm = hp.anafast(self.map, pol=False)
        cl = [clm]
        ax.plot(clm, 'k.', label='Data')
        for s in sims:
            self.loadSim(s)
            cls = hp.anafast(self.sim, pol=False)
            cl.append(cls)
            ax.plot(cls, '.', label='Sim '+str(s))
        ax.set_xlabel(r'$L$')
        ax.set_ylabel(r'$C_L$')
        ax.legend(loc='upper right', prop={'size':14})
        return cl
    
    def loadSim(self, n, plot=False):
        '''
        Loads the n-th simulation.
        
        !!!how do sim_lm give rise to 0 values in the area covered by mask??
        '''
        if self.res not in [256, 2048]:
            raise ValueError('Only res=256 or 2048 considered')
        
        if n==100:
            self.slm = self.alm
            self.sim = self.map
            return
        
        if self.phi:
            self.slm = np.load(self.dir+'_s'+STR2(n)+'_flm.npy')
            self.sim = np.load(self.dir+'_s'+STR2(n)+'_fmap.npy')
        else:
            self.slm = np.load(self.dir+'_s'+STR2(n)+'_klm.npy')
            self.sim = np.load(self.dir+'_s'+STR2(n)+'_kmap.npy')
        
        if self.conserv:
            hp.almxfl(self.slm, self.noiseFilt, inplace=True)
            
            cons_phi = '_fmapC.npy' if self.phi else '_kmapC.npy'
            cons_map = self.dir+'_s'+STR2(n)+cons_phi
            if os.path.isfile(cons_map):
                self.sim = np.load(cons_map)
            else:
                self.sim = hp.alm2map(self.slm, self.res, pol=False,
                                      verbose=False)
                np.save(cons_map, self.sim)
            
        if plot:
            Map = np.copy(self.sim)
            Map[self.mask==0.] = hp.UNSEEN
            Map = hp.ma(Map)
            if self.phi:
                ttl = r'Simulated lensing potential $\phi$'
                fmt = '%07.3e'
                unt = r'$\phi$'
            else:
                ttl = r'Simulated lensing convergence $\kappa$'
                fmt = '%.3f'
                unt = r'$\kappa$'
            cmap = ListedColormap(CMB_CMAP)
            cmap.set_under('w')
            cmap.set_bad('gray')
            hp.mollview(Map, title=ttl+' at $N_{side} = 2048$', 
                        format=fmt, cmap=cmap, cbar=True, unit=unt)

# Helper class
class BasicLensingMap(object):
    '''
    Writes relevant maps and alms so that they can be loaded from LensingMap
    class.
    '''
    def __init__(self, conserv, res):
        self.conserv = conserv
        
        if res is None:
            self.mask = hp.read_map(DIR+'data/aux/mask.fits', verbose=False)
            self.malm = hp.map2alm(self.mask)
            print('MASK')

            self.maskGal = hp.read_map(DIR+'data/aux/maskGal.fits', field=3, 
                                       verbose=False)
            self.malmGal = hp.map2alm(self.maskGal)
            print('MASKGAL')
            
            self.res = hp.npix2nside(self.mask.size)
            self.dir = DIR+'data/maps/n'+STR4(res)
            self.lmax = NSIDES[-1]
            
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
            
        elif res==2048:
            self.dir = DIR+'data/maps/n'+STR4(res)
            self.res = res
            self.lmax = LMAX(res)
            
            self.cb = np.load(self.dir+'a_cb.npy')
            self.lon = np.load(self.dir+'a_lon.npy')
            
            #!!! Downgrading affects noise correlations and thus nlkk and clkk
            self.nlkk = RAWSPEC[:self.lmax+1,1]
            self.clkk = RAWSPEC[:self.lmax+1,2] - RAWSPEC[:self.lmax+1,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.clff = fl**2 * self.clkk
            
            self.kmap = np.load(self.dir+'_kmap.npy')
            self.klm = np.load(self.dir+'_klm.npy')
            
            self.fmap = np.load(self.dir+'_fmap.npy')
            self.flm = np.load(self.dir+'_flm.npy')
            
            self.sim = None
            
            self.mask = np.load(self.dir+'a_mask.npy')
            self.malm = np.load(self.dir+'a_malm.npy')
            
            self.maskGal = np.load(self.dir+'a_maskGal.npy')
            self.malmGal = np.load(self.dir+'a_malmGal.npy')
            
        else:
            raise ValueError('Resolution (Nside) must be a power of 2')
    
    def set_res(self, res):
        '''
        Resets object with new resolution.
        '''
        self.__init__(phi=self.phi, conserv=self.conserv, res=res)
        
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
        
        hp.almxfl(self.klm, fl, inplace=True)
        hp.almxfl(self.flm, fl, inplace=True)
        hp.almxfl(self.malm, fl, inplace=True)
        hp.almxfl(self.malmGal, fl, inplace=True)
        
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
        if res!=NSIDES[-1]: self._lowRes(res)
        
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
            self.set_res(NSIDES[-1])

    def _writeSim(self, n):
        '''
        Produces output data files of given simulation at resolutions 256, 2048.
        !!! Only res = 256, 2048
        '''
        if self.res!=NSIDES[-1]: self.set_res(NSIDES[-1])
        
        sklm = hp.read_alm(DIR+'data/sims/obs_klms/sim_'+STR4(n)+'_klm.fits')
        fl = (2./(ell*(ell+1)) for ell in range(1, NSIDES[-1]+1))
        fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
        sflm = hp.almxfl(sklm, fl)
        
        print('HALF 2048')
        
        skmap = hp.alm2map(sklm, 2048, verbose=False)
        sfmap = hp.alm2map(sflm, 2048, verbose=False)
        
        np.save(DIR+'data/maps/n2048_s'+str(n).zfill(2)+'_klm', sklm)
        np.save(DIR+'data/maps/n2048_s'+str(n).zfill(2)+'_flm', sflm)
        np.save(DIR+'data/maps/n2048_s'+str(n).zfill(2)+'_kmap', skmap)
        np.save(DIR+'data/maps/n2048_s'+str(n).zfill(2)+'_fmap', sfmap)
        
        print('DONE 2048')
        
        res = 256
        lmax = LMAX(res)
        beam0 = hp.gauss_beam(FWHM[self.res], lmax)
        pixw0 = hp.pixwin(self.res)[:lmax+1]
        beam = hp.gauss_beam(FWHM[res], lmax)
        pixw = hp.pixwin(res)[:lmax+1]
        fl = (beam*pixw)/(beam0*pixw0)
        
        hp.almxfl(sklm, fl, inplace=True)
        hp.almxfl(sflm, fl, inplace=True)
        
        print('HALF 256')
        
        skmap = hp.alm2map(sklm, res, verbose=False)
        sfmap = hp.alm2map(sflm, res, verbose=False)
        
        np.save(DIR+'data/maps/n0256_s'+str(n).zfill(2)+'_klm', sklm)
        np.save(DIR+'data/maps/n0256_s'+str(n).zfill(2)+'_flm', sflm)
        np.save(DIR+'data/maps/n0256_s'+str(n).zfill(2)+'_kmap', skmap)
        np.save(DIR+'data/maps/n0256_s'+str(n).zfill(2)+'_fmap', sfmap)
        
        print('DONE 256')

    def _writeAllSim(self):
        '''
        Produces output data files of all simulations at resolutions 256, 2048.
        '''
        for n in range(NSIMS):
            print('SIM '+str(n))
            self._writeSim(n)






