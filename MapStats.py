'''
Analyses Cold Spot statistics

!!!
- smoothing of real data vs smoothing of sims
- Choose sim only if at least as extreme as CS (with S statistic)
- V is TOO low, M is high, S & K are TOO high (in abs values)
- Perform chi squared test
- Apply mask before calculations are performed
- Keep code fully independent of SMWH filter

?. Turn into lensing maps (signal to noise is high => simulations are/should be 
   noisy, units, transfer code, filters)
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from importlib import reload
import TempColdSpot as tcs
reload(tcs)

coordCS = (210, -57) #lon, lat in degrees in Galactic coordinates
titles = ('Mean', 'Variance', 'Skewness', 'Kurtosis') #All moments considered in
                                                      #the statistical analysis

class StatsMap(tcs.TempMap):
    """
    Contains all statistical analysis tools to detect local extrema on a map.
    """
    def __init__(self, res, aperture):
        '''
        Aperture determines the size of the disk within which local statistics
        are calculated
        '''
        tcs.TempMap.__init__(self, res)
        self.R = aperture
    
    def __repr__(self):
        return ("StatsMap(res = {0}, aperture = {1}) \n"
        ">>> map.size = {2}, lmax = {3}").format(self.res, self.R, 
                                          self.map.size, self.ELL.max())
    
    def set_R(self, aperture):
        self.R = aperture
    
    def detectCS(self, Map):
        '''
        Returns coordinates of coldest spot on given map. Coldest is defined by 
        lowest temperature.
        '''
        if Map is None: Map = self.map
        pix = np.where(Map==Map.min())[0][0]
        coord = hp.pix2ang(nside=self.res, ipix=pix)
        return coord

    def getDisk(self, centre):
        '''
        Returns pixels within the disk of given centre on any map, excluding 
        the boundaries.
        '''
        b, lon = centre
        VEC = hp.ang2vec(b, lon, lonlat=False)
        RADIUS = np.radians(self.R)
        pixs = hp.query_disc(nside=self.res, vec=VEC, radius=RADIUS, 
                             inclusive=False)
        return pixs

    def calcStats(self, Map, centre):
        '''
        Calculates the first four moments, starting from the mean, in given map
        and disk of given centre.
        '''
        pixs = self.getDisk(centre)
        if Map is None: Map = self.map
        sample = Map[pixs]
        
        N = pixs.size
        mean = 1./N * sample.sum()
        var  = np.sqrt( 1./N * (sample**2).sum() )
        skew = 1./(N*var**3) * (sample**3).sum()
        kur  = 1./(N*var**4) * (sample**4).sum() - 3 
        
        return mean, var, skew, kur

    def calcAvgStats(self, Map, centre):
        '''
        Calculates disk averages of the first four moments, as calculated by 
        calcStats().
        '''
        if Map is None: Map = self.map
        pixs = self.getDisk(centre)
        N = pixs.size
        moments = np.zeros(len(titles))
        for pix in pixs:
            coord = hp.pix2ang(nside=self.res, ipix=pix)
            stats = self.calcStats(Map, coord)
            moments += np.array(stats)
        return 1./N * moments
        
    def mapStats(self, Map, avg=True):
        '''
        Plots moments calculated at disks of class aperture, over the whole 
        given map.
        '''
        if Map is None: Map = self.map
        stats = [[] for i in range(len(titles))]
        
        for pix in range(Map.size):
            coord = hp.pix2ang(nside=self.res, ipix=pix)
            if avg: moments = self.calcAvgStats(Map, coord)
            if not avg: moments = self.calcStats(Map, coord)
            for i in range(len(titles)):
                stats[i].append(moments[i])
            
            if pix%1000==0: print('pix: ', pix)
        
        stats = np.array(stats)
        for i in range(len(titles)):
            hp.mollview(stats[i], title=titles[i])
        return stats

    def filtMap(self, Map):
        '''
        Applies SMHW filter on map. (Vielva, 2010)
        '''
        if Map is None:
            Map = self.map
            mlm = self.alm
        mlm = hp.map2alm(Map)
        
        cb, lon = hp.pix2ang(self.res, np.arange(Map.size))
        lmax = self.ELL.max()
        
        R = np.radians(self.R)
        W = mexHat(R, cb)
        
        wlm = hp.map2alm(W)[:lmax+1]
        ellFac = np.sqrt(4*np.pi/(2.*np.arange(lmax+1)+1))
        fl = ellFac * np.conj(wlm)
        
        convAlm = hp.almxfl(alm=mlm, fl=fl)
        newmap = hp.alm2map(convAlm, nside=self.res, pol=False, verbose=False)
        return newmap
    
    def filtMask(self, Mbd=0.9):
        '''
        Applies mask on selected pixels AFTER calculations have been performed.
        - Mbd: Only pixels of mask value >Mbd are considered before filtering
        (Zung, Huterer, 2010)
        '''
        cb, lon = hp.pix2ang(self.res, np.arange(self.map.size))
        lmax = self.ELL.max()
        
        mask = np.zeros(self.mask.size, dtype=float)
        mask[self.mask>Mbd] = self.mask[self.mask>Mbd]
        mlm = hp.map2alm(mask)
        
        R = np.radians(self.R)
        W = mexHat(R, cb)
        
        wwlm = hp.map2alm(W*W)[:lmax+1]
        ellFac = np.sqrt(4*np.pi/(2.*np.arange(lmax+1)+1))
        fl = ellFac*np.conj(wwlm)
        
        convAlm = hp.almxfl(alm=mlm, fl=fl)
        newmask = hp.alm2map(convAlm, nside=self.res, pol=False, verbose=False)
        
        return newmask

    def plotFilt(self, Map=None, mask=False, Mbd=0.9, Nbd=0.9):
        '''
        Plots given filtered map. Parameters include:
        - mask: if True, mask from filtMask() method is applied.
        - Mbd: Only pixels of mask value >Mbd are considered before filtering
        - Nbd: Only pixels of mask value >Nbd are considered after filtering
        '''
        Map = self.filtMap(Map)
        if mask:
            mask = self.filtMask(Mbd)
            Map[mask<Nbd] = hp.UNSEEN
            Map = hp.ma(Map)
        hp.mollview(Map, title='Filtered CMB T', cbar=True, unit=r'$K$')

    def compareSims(self, nsims=100):
        '''
        Calculates moments of real map and of nsims in number simulations.
        Moments are based on disk averages. 
        '''
        data = self.filtMap(Map=None)
        coord = lonlat2colatlon(coordCS)
        moments = self.calcAvgStats(Map=data, centre=coord)
        for s in range(nsims):
            sim = self.genSim(plot=False, mask=False)
            sim = self.filtMap(sim)
            coord = self.detectCS(sim)
            newmoments = self.calcAvgStats(sim, coord)
            moments = np.vstack((moments, newmoments))
            if s%10==0: print('sim: ', s)
        return moments
    
    def histSim(self, nsims=100, bins=10, normed=False):
        '''
        Produces histogram of given number of simulations against real data for
        all four moments calculated with given disk apperture. Parameters bins 
        and normed affect illustration of the histogram.
        '''
        moments = self.compareSims(nsims)
        data = moments[0]
        sims = moments[1:]
        
        fig = plt.figure()
        c = ('b', 'r', 'y', 'g')
        for i in range(len(titles)):
            ax = fig.add_subplot(2,2,i+1)
            ax.hist(sims[:,i], bins=bins, normed=normed, color=c[i])
            ax.axvline(x=data[i], color='k', ls='--')
            ax.set_xlabel(titles[i])
            ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.tight_layout()
    
    def genAngProfiles(self, nsims=100, apertures=[5], plot=False):
        '''
        Plots angluar profiles of all four moments of real data against the 
        average of given number of simulations.
        '''
        self.set_R(apertures[0])
        print('R: ', self.R)
        moments = self.compareSims(nsims=nsims)
        for aperture in apertures[1:]:
            self.set_R(aperture)
            print('R: ', self.R)
            newmoments = self.compareSims(nsims)
            moments = np.dstack((moments, newmoments))
        data = moments[0]
        simsAvg = moments[1:].mean(axis=0)
        simsStd = moments[1:].std(axis=0)
        
        if plot:
            fig = plt.figure()
            for i in range(len(titles)):
                ax = fig.add_subplot(2,2,i+1)
                ax.plot(apertures, simsAvg[i], 'k--')
                ax.plot(apertures, data[i], 'rx')
                ax.fill_between(apertures, simsAvg[i] - simsStd[i],
                  simsAvg[i] + simsStd[i], alpha=0.4, facecolor='darkslategray')
                ax.fill_between(apertures, simsAvg[i] - 2*simsStd[i],
                  simsAvg[i] + 2*simsStd[i], alpha=0.2, facecolor='slategrey')
                ax.set_xlabel(r'Aperture (deg)')
                ax.set_ylabel(titles[i])
                ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
            fig.tight_layout()
        
        return np.dstack((data, simsAvg, simsStd))

def lonlat2colatlon(coord):
    '''
    - coord: tuple in form (longitude, latitude)
    Returns tuple in form (colatitude, longitude)
    '''
    lon, lat = coord
    b, lon = np.radians(90-lat), np.radians(lon)
    return b, lon

def mexHat(R, cb):
    '''
    Computes SMHW function for scale R and array of co-latitudes cb.
    '''
    # Transformation of variable - ignores conventional factor of 2
    y = np.tan(cb/2.)
    
    # Squares
    yy = y*y
    RR = R*R
    
    # Normalisation coefficient for square of wavelet
    A = 1./np.sqrt(2*np.pi*RR*(1. + RR/2. + RR*RR/4.))
    
    # Wavelet function
    W = A * (1. + yy)*(1. + yy) * (2. - 4./RR * yy) * np.exp(-2./RR * yy)
    
    return W


'''
# TESTING FUNCS

from scipy.special import sph_harm as sphh
def WC(tmap, R=2): # Not-working alternative to SMHW filter
    T = tmap.map
    cb, lon = hp.pix2ang(tmap.res, np.arange(T.size))
    R = np.radians(R)
    lmax = tmap.cl.size-1
    
    tlm = tmap.alm
    Psi = mexHat(R, cb)
    
    w = np.zeros(cb.size, dtype='complex128')
    for ell in range(lmax+1):
        for em in range(ell+1):
            if ell in [50,60,70,80,90]: print('lm: ', ell, em)
            if True in np.isnan(w):
                print('NANERROR at lm: ', ell, em)
                raise ValueError
            if em==0: 
                inc = tlm[np.where(tmap.ELL==ell)][em]*sphh(em, ell, lon, cb)
            else:
                Y = sphh(em, ell, lon, cb)
                if np.abs(Y[0])<1.e-40 or np.isnan(np.abs(Y[0])):
                    inc = 0
                else:
                    Yn = sphh(-em, ell, lon, cb)
                    inc = tlm[np.where(tmap.ELL==ell)][em]*(Y + Yn)
            w +=inc
    w = Psi * T * w
    return w

def convGauss(tmap, sigma=0.01): # Alternative to hp.smoothing method
    lmax = tmap.cl.size - 1
    ell = np.arange(lmax+1)

    ellFac = np.exp(-0.5 * ell * (ell + 1) * sigma ** 2)
    tlm = tmap.alm
    
    tlmConv = hp.almxfl(alm=tlm, fl=ellFac)
    tmapConv = hp.alm2map(tlmConv, nside=tmap.res, pol=False, verbose=False)
    tmapConvIdeal = hp.smoothing(tmap.map, sigma=sigma)
    
    hp.mollview(tmapConvIdeal, title='ReadyMeal')
    hp.mollview(tmapConv, title='CookedChicken')
    return tmapConvIdeal, tmapConv

def filtMap(self): # Alternative to filtMap method calculation
    wlm = hp.map2alm(W)
    for i in range(3*self.res):
        wlm[np.where(self.ELL==self.ELL[i])] = wlm[i]
    ellfac = np.sqrt(4*np.pi/(2*self.ELL+1))
    
    convAlm = ellfac * np.conj(wlm) * self.alm
    newmap = hp.alm2map(alms=convAlm, nside=self.res, pol=False, 
                        verbose=False)
    
def testMask(self, Mbd, Nbd): # Tests mask
    a,b,c = self.filtMask(Mbd=Mbd, Nbd=Nbd)
    self.plotMap(mask=True)
    sim = self.genSim(plot=False, mask=False)
    filtMap = self.filtMap(sim)
    filtMap[c<Nbd] = hp.UNSEEN
    filtMap = hp.ma(filtMap)
    hp.mollview(filtMap, title='Filtered CMB T', cbar=True, unit=r'$K$')
    return self.mask, c
    
'''




































