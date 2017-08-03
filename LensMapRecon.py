"""
Reconstruction of CMB Lensing Map from Planck 2015 data release.

!!! FIX:
- plot spectrum method, is spectrum invariant under downgrading?
- fix mask filtering
- simulations
- docstrings
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
STR = lambda res: str(res).zfill(4)

###

class LensingMap(object):
    """
    Represents the CMB Lensing Map of the Planck 2015 data release (ref. Planck
    2015 results XV. Gravitational lensing).
    """
    def __init__(self, res=None):
        '''        
        - mask: lensing potential map mask #!!! reference
        - klm: spherical harmonic coefficients of lensing convergence kappa
        - rawSpec: approximate noise and signal+noise power spectra of kappa
        '''
        self.dir = 'CMBL_Maps/'
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
            
            self.nlkk = rawSpec[:self.lmax+2,1]
            self.clkk = rawSpec[:self.lmax+2,2] - rawSpec[:self.lmax+2,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.flm = hp.almxfl(self.klm, fl)
            self.fmap = hp.alm2map(self.flm, self.res, verbose=False)
            self.clff = fl**2 * self.clkk
            print('FMAP')
            
        elif res in NSIDES:
            core = self.dir + 'data/n' + STR(res)
            self.res = res
            self.lmax = LMAX(res)
            
            self.nlkk = rawSpec[:self.lmax+1,1]
            self.clkk = rawSpec[:self.lmax+1,2] - rawSpec[:self.lmax+1,1]
            
            fl = (2./(ell*(ell+1)) for ell in range(1, self.lmax+1))
            fl = np.concatenate( (np.ones(1), np.fromiter(fl, np.float64)) )
            self.clff = fl**2 * self.clkk
            print('KMAP')
            self.kmap = hp.read_map(core+'_kmap.fits', verbose=False)
            self.klm = hp.read_alm(core+'_klm.fits')
            print('KMAP -> FMAP')
            
            self.fmap = hp.read_map(core+'_fmap.fits', verbose=False)
            self.flm = hp.read_alm(core+'_flm.fits')
            print('FMAP -> MASK')
            
            self.mask = hp.read_map(core+'_mask.fits', verbose=False)
            self.malm = hp.read_alm(core+'_malm.fits')
            print('MASK -> MASKGAL')
            
            self.maskGal = hp.read_map(core+'_maskGal.fits', verbose=False)
            self.malmGal = hp.read_alm(core+'_malmGal.fits')
            
            self.sim = None
            
        else:
            raise ValueError('Resolution (Nside) must be a power of 2')
    
    def set_res(self, res):
        '''
        Resets object with new resolution.
        '''
        self.__init__(int(res))
        
    def _lowRes(self, res, bd=0.9):
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
        core = self.dir + 'data/n' + STR(res)
        if res!=2048: self._lowRes(res)
        
        hp.write_map(core+'_kmap.fits', self.kmap, nest=False)
        hp.write_alm(core+'_klm.fits', self.klm)
        
        hp.write_map(core+'_fmap.fits', self.fmap, nest=False)
        hp.write_alm(core+'_flm.fits', self.flm)
        
        hp.write_map(core+'_mask.fits', self.mask, nest=False)
        hp.write_alm(core+'_malm.fits', self.malm)
        
        hp.write_map(core+'_maskGal.fits', self.maskGal, nest=False)
        hp.write_alm(core+'_malmGal.fits', self.malmGal)
    
    def _writeAll(self):
        for res in NSIDES[:-1]:
            print(res)
            self.set_res(NSIDES[-1])
            self._write(res)
    
    def plotMap(self, phi=True, mask=False):
        '''
        Plots map including mask if True.
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
        hp.mollview(Map, coord='G', title=title, cbar=True, 
                    unit=r'dimensionless')
    
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
        
"""
Possible additional methods:

    def lensingMap(self, phi=True, plot=True):
        '''
        Returns the lensing convergence.
        
        - phi: if True, returns the lensing potential instead of the lensing 
               convergence kappa
        - plot: if True, also plots data on a map with Mollweide projection
        '''
        Map = self.klm
        
        if plot:
            title = r'Lensing convergence $\kappa$'
            Plot = Map
            Plot[self.mask==0.] = hp.UNSEEN
            Plot = hp.ma(Plot)
            hp.mollview(Map, title=title, cbar=True, unit='no units') #Weyl Psi
        
        self.map = Map
        return Map
    
    def powerSpectrum(self, plot=True):
        '''
        Returns the kappa power spectrum.
        
        - plot: if True, also plots the spectrum
        #!!! add error bars (aggressive binning)
        '''        
        if plot:
            y = 2./np.pi * 1e7 * self.clkk
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.semilogx(ell, y)
            ax.set_xlabel(r'$L$')  
            ax.set_ylabel(r'$\frac{[L(L+1)]^2}{2\pi} C_L^{\phi \phi}\ [\times 10^7]$')
    
    def calcCl(self, noise=True, phi=True):
        '''
        Calculates averaged spherical harmonic coefficients C_l for kappa.
        
        - noise: if True, keeps noise in calculations
        - phi: if True, calculates for phi instead
        #!!! fix docstring
        '''            
        if noise and phi:
            self.lensingMap(phi=True, plot=False)
            cl = hp.alm2cl(self.flm)
            self.clff = cl
        elif noise and not phi:
            cl = hp.alm2cl(self.klm)
            self.clkk = cl
        elif not noise and phi:
            self.powerSpectrum(plot=False)
            ell = np.arange(self.NSIDE+1); ell[0]=1
            cl = 4./(ell*(ell+1.))**2*self.clkk
            self.clkk = cl
        elif not noise and not phi:
            self.powerSpectrum(plot=False)
            cl = self.clkk
        return cl

NSIDE = 2048

def spectrum_clf():
    '''
    '''
    klm15 = hp.read_alm('data/dat_klm.fits')
    
    ell, em = hp.Alm.getlm(NSIDE)
    flm15 = 2./(ell[1:]*(ell[1:]+1))*klm15[1:]
    flm15 = np.concatenate((np.array([0]), flm15))
    clf = hp.alm2cl(flm15)
    
#    clftest = []
#    for el in range(NSIDE+1):
#        run = 0
#        for em in range(el):
#            idx = hp.Alm.getidx(NSIDE, el, em)
#            run += np.absolute(flm15[idx])**2
#        clftest.append(2./(2.*el+1) * run)
#    clftest = np.array(clftest)
            
    
    elly = np.arange(NSIDE+1)
    coefy = ( ( elly*(elly+1.) )**2/(2.*np.pi) )*1.e7
    y = coefy*clf
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(elly, y)
    ax.set_title('cl f')
    return clf

def wiener_filter(map15):
    '''
    Band-limited between (and including) 8-2048.
    '''
    spec = np.loadtxt("data/nlkk.dat")
    fltr = spec[:,0::2]
    fltr[:,1] = (fltr[:,1] - spec[:,1]) / spec[:,2]
    print(fltr[0,0], fltr[0,1])
        
    for el in range(8,NSIDE+1):
        for em in range(el+1):
            idx = hp.Alm.getidx(NSIDE, el, em)
            map15[idx] *= fltr[el-8,1]
            if el==9:
                print(em)
                print(hp.Alm.getlm(NSIDE, idx))
        
    return map15

def compareMaps():
    klm15 = hp.read_alm('data/dat_klm.fits')
    spectrum = np.loadtxt('data/nlkk.dat')
    ell = spectrum[:,0]
    noise = spectrum[:,1]
    power = spectrum[:,2] - spectrum[:,1]
    
    map15og = hp.alm2map(klm15, nside=NSIDE)
    map15 = hp.synfast(np.concatenate((np.zeros(8), power)), NSIDE)
    
    hp.mollview(map15og)    
    hp.mollview(map15)    
    return map15og, map15

"""
















