"""
Reconstruction of CMB Lensing Map from Planck 2015 data release.

beam fwhm = 0.00145444 rad (5') ->
five lensing potential estimators (Okamoto & Hu) ->

"""
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt

class LensingMap(object):
    """
    Represents the CMB Lensing Map of the Planck 2015 data release (ref. Planck
    2015 results XV. Gravitational lensing).
    """
    def __init__(self):
        '''
        Checks for Planck data at the expected directory ('/home2/nkoukou/data')
        and then imports:
        
        - mask: lensing potential map mask #!!! reference
        - klm: spherical harmonic coefficients of lensing convergence kappa #!!! units?
        - rawSpec: approximate noise and signal+noise power spectra of kappa #!!! units?
        '''
        try:
            self.mask = hp.read_map('data/mask.fits')
            self.klm = hp.read_alm('data/dat_klm.fits')
            self.rawSpec = np.loadtxt('data/nlkk.dat')
        except:
            print('Files are not in expected directory')
            
        self.NSIDE = 2048 # !!! refer to header/readme
        lm = hp.Alm.getlm(self.NSIDE)
        self.ELL = lm[0]
        self.EM = lm[1]
        
        self.map = None
        self.flm = None
        self.clkk = None
        self.clff = None
        self.nlkk = None

    def lensingMap(self, phi=True, plot=True):
        '''
        Returns the lensing convergence.
        
        - phi: if True, returns the lensing potential instead of the lensing 
               convergence kappa
        - plot: if True, also plots data on a map with Mollweide projection
        '''
        if phi:
            flm = 2./(self.ELL[1:]*(self.ELL[1:]+1.))*self.klm[1:]
            flm = np.concatenate((np.array([0]), flm)) #!!! purify division by 0
            Map = hp.alm2map(flm, nside=self.NSIDE)
            title = r'Lensing potential $\phi$'
            self.flm = flm
        elif not phi:
            Map = hp.alm2map(self.klm, nside=self.NSIDE)
            title = r'Lensing convergence $\kappa$'
        
        if plot:
            Plot = Map
            Plot[self.mask==0.] = hp.UNSEEN
            Plot = hp.ma(Plot)
            hp.mollview(Map, title=title, cbar=True, unit='dimensionless?')
        
        self.map = Map
        return Map
    
    def powerSpectrum(self, plot=True):
        '''
        Returns the kappa power spectrum.
        
        - plot: if True, also plots the spectrum
        #!!! add error bars (aggressive binning)
        '''
        ell = self.rawSpec[:,0]
        nlkk = self.rawSpec[:,1]
        clkk = self.rawSpec[:,2] - self.rawSpec[:,1]
        self.clkk = np.concatenate((np.zeros(8), clkk))
        self.nlkk = np.concatenate((np.zeros(8), nlkk))
        
        if plot:
            y = 2./np.pi * 1e7 * clkk
            fig = plt.figure()
            ax = fig.add_subplot(211)
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
                
"""
Possible additional methods:


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
















