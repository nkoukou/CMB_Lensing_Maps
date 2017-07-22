'''
Analyses Cold Spot statistics

!!!
- Apply mask
- Check almxfl method
- Check filter
- Choose sim only if at least as extreme as CS
- V is TOO low, M is high, S & K are TOO high (in abs values)
- Perform chi squared test
- Comment module

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
titles = ('Mean', 'Variance', 'Skewness', 'Kurtosis')

class StatsMap(tcs.TempMap):
    """
    Contains all statistical analysis tools to detect local extrema on a map.
    """
    def detectCS(self, Map=None):
        if Map is None: Map = self.map
        pix = np.where(Map==Map.min())[0][0]
        coord = hp.pix2ang(nside=self.res, ipix=pix)
        return coord

    def getDisk(self, centre, aperture):
        b, lon = centre
        VEC = hp.ang2vec(b, lon, lonlat=False)
        RADIUS = np.radians(aperture)
        pixs = hp.query_disc(nside=self.res, vec=VEC, radius=RADIUS, 
                             inclusive=False)
        return pixs

    def calcStats(self, centre, aperture):
        pixs = self.getDisk(centre, aperture)
        sample = self.map[pixs]
        
        N = pixs.size
        mean = 1./N * sample.sum()
        var  = np.sqrt( 1./N * (sample**2).sum() )
        skew = 1./(N*var**3) * (sample**3).sum()
        kur  = 1./(N*var**4) * (sample**4).sum() - 3 
        
        return mean, var, skew, kur

    def calcAvgStats(self, centre, aperture):
        pixs = self.getDisk(centre, aperture)
        N = pixs.size
        moments = np.zeros(len(titles))
        for pix in pixs:
            coord = hp.pix2ang(nside=self.res, ipix=pix)
            stats = self.calcStats(centre=coord, aperture=aperture)
            moments += np.array(stats)
        return 1./N * moments
        
    def mapStats(self, aperture, avg=True):
        stats = [[] for i in range(len(titles))]
        
        for pix in range(self.map.size):
            coord = hp.pix2ang(nside=self.res, ipix=pix)
            if avg: moments = calcAvgStats(coord, aperture)
            elif not avg: moments = calcStats(coord, aperture)
            for i in range(len(titles)):
                stats[i].append(moments[i])
            
            if pix%1000==0: print('pix:', pix)
        
        stats = np.array(stats)
        for i in range(len(titles)):
            hp.mollview(stats[i], title=titles[i])
        return stats

    def filtMap(self, aperture=2, sim=False):
        if sim:
            Map = self.genSim()
            mlm = hp.map2alm(Map)
        elif not sim:
            Map = self.map
            mlm = self.alm
        cb, lon = hp.pix2ang(self.res, np.arange(Map.size))
        lmax = self.ELL.max()
        
        R = np.radians(aperture)
        W = mexHat(R, cb)
        
        '''
        wlm = hp.map2alm(W)[:lmax+1]
        ellFac = np.sqrt(4*np.pi/(2*np.arange(lmax+1)+1))
        fl = ellFac * np.conj(wlm)
        
        convAlm = hp.almxfl(alm=mlm, fl=fl, mmax=False)
        newmap = hp.alm2map(alms=convAlm, nside=self.res, pol=False, 
                            verbose=False)
        '''
        wlm = hp.map2alm(W)
        for i in range(3*self.res):
            wlm[np.where(self.ELL==self.ELL[i])] = wlm[i]
        ellfac = np.sqrt(4*np.pi/(2*self.ELL+1))
        
        convAlm = ellfac * np.conj(wlm) * self.alm
        newmap = hp.alm2map(alms=convAlm, nside=self.res, pol=False, 
                            verbose=False)
        return newmap

    def compareSims(self, nsims=100, aperture=2):
        data = self.filtMap(aperture=aperture, sim=False)
        coord = lonlat2colatlon(coordCS)
        moments = self.calcAvgStats(centre=coord, aperture=aperture)
        for s in range(nsims):
            sim = self.filtMap(aperture=aperture, sim=True)
            coord = self.detectCS(Map=sim)
            newmoments = self.calcAvgStats(centre=coord, aperture=aperture)
            moments = np.vstack((moments, newmoments))
            if s%10==0: print('sim: ', s)
        return moments
    
    def histSim(self, nsims=100, aperture=2, bins=10, normed=False):
        moments = self.compareSims(nsims=nsims, aperture=aperture)
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
    
    def genAngProfiles(self, nsims=100, apertures=np.linspace(1, 25, 9), plot=False):
        print('R: ', apertures[0])
        moments = self.compareSims(nsims=nsims, aperture=apertures[0])
        for aperture in apertures[1:]:
            print('R: ', aperture)
            newmoments = self.compareSims(nsims=nsims, aperture=aperture)
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









































