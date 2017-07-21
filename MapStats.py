'''
Analyses Cold Spot statistics
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
    def detectCS(self):
        pix = np.where(self.map==self.map.min())[0]
        coord = hp.pix2ang(nside=self.res, ipix=pix)
        return coord

    def calcStats(self, centre, aperture):
        pixs = getDisk(centre, aperture)
        sample = self.map[pixs]
        
        N = pixs.size
        mean = 1./N * sample.sum()
        var  = np.sqrt( 1./N * (sample**2).sum() )
        skew = 1./(N*var**3) * (sample**3).sum()
        kur  = 1./(N*var**4) * (sample**4).sum() - 3 
        
        return mean, var, skew, kur

    def calcAvgStats(self, centre, aperture):
        pixs = getDisk(centre, aperture)
        N = pixs.size
        moments = np.zeros(len(titles))
        for pix in pixs:
            coord = hp.pix2ang(nside=self.res, ipix=pix)
            stats = calcStats(coord, aperture)
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

    def filtMap(self, aperture=2):
        cb, lon = hp.pix2ang(self.res, np.arange(self.map.size))
        
        R = np.radians(aperture)
        W = mexHat(R, cb)
        
        wlm = hp.map2alm(W)
        for i in range(3*self.res):
            wlm[np.where(self.ELL==self.ELL[i])] = wlm[i]
        ellfac = np.sqrt(4*np.pi/(2*self.ELL+1))
        
        convAlm = ellfac * np.conj(wlm) * self.alm
        newmap = hp.alm2map(convAlm, self.res)
        return newmap

    def filtSim(self, aperture=2):
        sim = self.genSim()
        cb, lon = hp.pix2ang(self.res, np.arange(sim.size))
        
        R = np.radians(aperture)
        W = mexHat(R, cb)
        
        wlm = hp.map2alm(W)
        for i in range(3*self.res):
            wlm[np.where(self.ELL==self.ELL[i])] = wlm[i]
        ellfac = np.sqrt(4*np.pi/(2*self.ELL+1))
        mlm = hp.map2alm(sim)
        
        convAlm = ellfac * np.conj(wlm) * mlm
        newmap = hp.alm2map(convAlm, self.res)
        return newmap


def lonlat2colatlon(coord):
    '''
    - coord: tuple in form (longitude, latitude)
    Returns tuple in form (colatitude, longitude)
    '''
    lon, lat = coord
    b, lon = np.radians(90-lat), np.radians(lon)
    return b, lon

def getDisk(centre, aperture):
    b, lon = centre
    VEC = hp.ang2vec(b, lon, lonlat=False)
    RADIUS = np.radians(aperture)
    pixs = hp.query_disc(nside=NSIDE, vec=VEC, radius=RADIUS, inclusive=False)
    return pixs

def mexHat(R, cb):
    '''
    Computes SMHW function for scale R and array of co-latitudes cb.
    '''
    # Transformation of variable - ignores conventional factor of 2
    y = np.tan(cb/2.)
    
    # Normalisation coefficient for square of wavelet
    A = 1./( R*np.sqrt(2*np.pi*(1. + R**2/2. + R**4/4.)) )
    
    # Wavelet function
    W = A * (1 + y*y)**2 * (2 - y*y) * np.exp(-2./(R*R) * y*y)
    
    return W









































