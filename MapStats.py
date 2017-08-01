'''
Analyses Cold Spot statistics.

!!!
- Cleaner code, more independent classes
- Methods should depend on lmax
- Fix mask according to 2013 XXIII
- V is TOO low, M is high, S & K are TOO high (in abs values)
- Perform chi squared test

?. Turn into lensing maps (signal to noise is high => simulations are/should be 
   noisy, units, transfer code, filters)
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from importlib import reload #!!!
import TempColdSpot as tcs
import MapFilts as mf
reload(tcs)
reload(mf)

# Global constants and functions
MOMENTS = ('Mean', 'Variance', 'Skewness', 'Kurtosis') #All moments considered 
                                                       #in the analysis

MAP = tcs.TempMap(2048)
FILT = mf.FilterMap(MAP.res, 80, 5, 0)

FMASK = np.load(MAP.dir+tcs.STR(MAP.res)+'e_fmask.npy')

def lonlat2colatlon(coord):
    '''
    - coord: tuple in form (longitude, latitude)
    Returns tuple in form (colatitude, longitude)
    '''
    lon, lat = coord
    cb, lon = np.radians(90-lat), np.radians(lon)
    return cb, lon

def colatlon2lonlat(coord):
    '''
    - coord: tuple in form (longitude, latitude)
    Returns tuple in form (colatitude, longitude)
    '''
    cb, lon = coord
    lon, lat = np.rad2deg(lon), 90-np.rad2deg(cb)
    if isinstance(lon, float):
        if lon>180: lon -=360
    else:
        if lon[0]>180.: lon -=360
    return lon, lat

###

def detectCS(Map, mask=FMASK):
    '''
    Returns coordinates of coldest spot on given map. Coldest is defined by 
    lowest temperature.
    '''
    pix = np.where(Map==Map[mask==1].min())[0][0]
    coord = hp.pix2ang(nside=MAP.res, ipix=pix)
    return coord

def getDisk(centre, radius, mask=MAP.mask):
    '''
    Returns pixels within the disk of given centre on any map, excluding 
    the boundaries. Only unmasked pixels by given mask are returned.
    '''
    cb, lon = centre
    VEC = hp.ang2vec(cb, lon, lonlat=False)
    pixs = hp.query_disc(MAP.res, vec=VEC, radius=radius, inclusive=False)
    pixs = pixs[np.where(mask[pixs]==1.)]
    fail = pixs[np.where(mask[pixs]==0.)]
    return pixs, fail

def calcStats(centre, Map, mask=MAP.mask):
    '''
    Calculates the first four moments, starting from the mean, in given map
    and disk of given centre and radius. Only unmasked pixels by given mask are 
    considered.
    '''
    pixs = getDisk(centre, FILT.R, mask)
    sample = Map[pixs]
    
    N = sample.size
    mean = 1./N * sample.sum()
    var  = np.sqrt( 1./N * (sample**2).sum() )
    skew = 1./(N*var**3) * (sample**3).sum()
    kur  = 1./(N*var**4) * (sample**4).sum() - 3 
    
    return pixs, mean, var#, skew, kur

def compareTemp(nsims=100, plot=True):
    '''
    Calculates moments of real map and of nsims in number simulations.
    Moments are based on disk averages. 
    '''
    data = FILT.filterMap(MAP, MAP.map)
    coord = lonlat2colatlon(tcs.COORDCS)
    pixs = getDisk(coord, FILT.R, FMASK)
    TCS = data[pixs].min()
    
    T = np.zeros(nsims)
    for s in range(nsims):
        if s%1==0: print('sim: ', s)
        sim = MAP.genSim(lmax=FILT.lmax)
        sim = FILT.filterMap(MAP, sim)
        T[s] = sim[FMASK==1].min()
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(T, bins=20, normed=False, color='b')
        ax.axvline(x=TCS, color='k', ls='--')
        ax.set_xlabel(r'$T_{cold}$')
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    return np.concatenate((np.array([TCS]), T))

def compareSims(nsims=100, plot=True, bins=10, normed=False):
    '''
    Calculates moments of real map and of nsims in number simulations.
    Moments are based on disk averages. 
    '''
    data = FILT.filterMap(MAP, MAP.map)
    coord = lonlat2colatlon(tcs.COORDCS)
    moments = calcAvgStats(coord, data)
    for s in range(nsims):
        sim = MAP.genSim(lmax=FILT.lmax)
        sim = FILT.filterMap(MAP, sim)
        coord = detectCS(sim)
        newmoments = calcAvgStats(coord, sim)
        moments = np.vstack((moments, newmoments))
        if s%10==0: print('sim: ', s)
    
    if plot:
        data = moments[0]
        sims = moments[1:]
        
        fig = plt.figure()
        c = ('b', 'r', 'y', 'g')
        for i in range(len(MOMENTS)):
            ax = fig.add_subplot(2,2,i+1)
            ax.hist(sims[:,i], bins=bins, normed=normed, color=c[i])
            ax.axvline(x=data[i], color='k', ls='--')
            ax.set_xlabel(MOMENTS[i])
            ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.tight_layout()
    return moments

def genAngProfiles(nsims=100, apertures=[5], plot=False):
    '''
    Plots angluar profiles of all four moments of real data against the 
    average of given number of simulations.
    '''
    Rf = FILT.R
    FILT.set_R(apertures[0])
    print('R: ', FILT.R)
    moments = compareSims(nsims=nsims)
    for aperture in apertures[1:]:
        FILT.set_R(aperture)
        print('R: ', FILT.R)
        newmoments = compareSims(nsims)
        moments = np.dstack((moments, newmoments))
    data = moments[0]
    simsAvg = moments[1:].mean(axis=0)
    simsStd = moments[1:].std(axis=0)
    
    if plot:
        fig = plt.figure()
        for i in range(len(MOMENTS)):
            ax = fig.add_subplot(2,2,i+1)
            ax.plot(apertures, simsAvg[i], 'k--')
            ax.plot(apertures, data[i], 'rx')
            ax.fill_between(apertures, simsAvg[i] - simsStd[i],
              simsAvg[i] + simsStd[i], alpha=0.4, facecolor='darkslategray')
            ax.fill_between(apertures, simsAvg[i] - 2*simsStd[i],
              simsAvg[i] + 2*simsStd[i], alpha=0.2, facecolor='slategrey')
            ax.set_xlabel(r'Aperture (deg)')
            ax.set_ylabel(MOMENTS[i])
            ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.tight_layout()
    
    Rf = int(np.rad2deg(Rf))
    FILT.set_R(Rf)
    return np.dstack((data, simsAvg, simsStd))

def calcArea(nsims=100, thresh=3, mask=FMASK):
    data = FILT.filterMap(MAP, MAP.map)
    for pix in range(data.size):
        if mask[pix]==0.: continue
        coord = hp.pix2ang(nside=MAP.res, ipix=pix)
        pixs = getDisk(centre=coord, radius=FILT.R, mask=mask)
        sample = data[pixs]
        var  = np.sqrt( 1./N * (sample**2).sum() )
        for s in sample:
            pass #if s>thresh
        
        
'''
# TESTING FUNCS
    
def mapStats(Map, avg=True, mask=MAP.mask):
    """
    Plots moments calculated at disks of class aperture, over the whole 
    given map. Only unmasked pixels by given mask are considered.
    """
    stats = [[] for i in range(len(MOMENTS))]
    for pix in range(Map.size):
        if mask[pix]==0.:
            for i in range(len(MOMENTS)):
                stats[i].append(hp.UNSEEN)
            continue
        coord = hp.pix2ang(nside=MAP.res, ipix=pix)
        if avg: moments = calcAvgStats(coord, Map, mask)
        if not avg: moments = calcStats(coord, Map, mask)
        for i in range(len(MOMENTS)):
            stats[i].append(moments[i])
        
        if pix%1000==0: print('pix: ', pix)
    
    stats = np.array(stats)
    for i in range(len(MOMENTS)):
        stat = hp.ma(np.copy(stats[i]))
        hp.mollview(stat, title=MOMENTS[i])
    return stats

def calcAvgStats(centre, Map, mask=MAP.mask):
    """
    Calculates disk averages of the first four moments, as calculated by 
    calcStats(). Only unmasked pixels by given mask are considered.
    """
    pixs = getDisk(centre, FILT.R, mask)
    N = pixs.size
    moments = np.zeros(len(MOMENTS))
    for pix in pixs:
        coord = hp.pix2ang(nside=MAP.res, ipix=pix)
        stats = calcStats(coord, Map, mask)
        moments += np.array(stats)
    return 1./N * moments

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


def filtMask(self, Mbd=0.9):
    """
    Applies mask on selected pixels AFTER calculations have been performed.
    - Mbd: Only pixels of mask value >Mbd are considered before filtering
    (Zung, Huterer, 2010)
    """
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

def plotFilt(self, Map=None, mask=False, Mbd=0.9, Nbd=0.9):
    """
    Plots given filtered map. Parameters include:
    - mask: if True, mask from filtMask() method is applied.
    - Mbd: Only pixels of mask value >Mbd are considered before filtering
    - Nbd: Only pixels of mask value >Nbd are considered after filtering
    """
    Map = self.filtMap(Map)
    if mask:
        mask = self.filtMask(Mbd)
        #Map[mask<Nbd] = hp.UNSEEN
        Map[self.mask==0] = hp.UNSEEN
        Map = hp.ma(Map)
    hp.mollview(Map, title='Filtered CMB T (scale={0})'.format(self.R), 
                cbar=True, unit=r'$K$')

def pickGalMask(mask):
    nside = hp.npix2nside(mask.size)
    galaxy = [hp.ang2pix(2048, np.pi/2, 0)]
    s = 0
    thresh = int(5e6)
    
    while len(galaxy)<thresh:
        print(len(galaxy))
        for g in galaxy[s:]:
            nn = hp.get_all_neighbours(nside, g)
            for n in nn:
                if n in galaxy: continue
                if mask[n]==0: 
                    galaxy.append(n)
            s +=1
    return np.array(galaxy)

def pickGalMask(mask):
    mid = int(mask.size/2)
    q = mid
    cond = True
    while cond or mask[q]==0:
        if q%int(5e5)==0: print(q)
        if mask[q]!=0:
            test = []
            for n in range(1000):
                test.append(mask[q-n])
                if 0 not in test:
                    cond = False
                else:
                    cond = True
                    q -= np.where(np.array(test)==0)[0][0]
        q -=1
    return q
'''




































