'''
Analyses Cold Spot statistics.
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from importlib import reload #!!!
import TempColdSpot as tcs
from MapFilts import filterMap

# Global constants and functions
MOMENTS = ('Mean', 'Variance', 'Skewness', 'Kurtosis') #All moments considered 
                                                       #in the analysis
LMAX = 80
MAP = tcs.TempMap(128)

###

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

def detectCS(Map, mask):
    '''
    Returns coordinates of coldest spot on given map. Coldest is defined by 
    lowest temperature.
    '''
    pix = np.where(Map==Map[mask==1].min())[0][0]
    coord = hp.pix2ang(nside=MAP.res, ipix=pix)
    return coord

def getDisk(centre, radius, mask):
    '''
    Returns pixels within the disk of given centre on any map, excluding 
    the boundaries. Only unmasked pixels by given mask are returned.
    '''
    cb, lon = centre
    VEC = hp.ang2vec(cb, lon, lonlat=False)
    pixs = hp.query_disc(MAP.res, vec=VEC, radius=radius, inclusive=False)
    pixs = pixs[np.where(mask[pixs]==1.)]
    return pixs

def calcStats(centre, radius, Map, mask):
    '''
    Calculates the first four moments, starting from the mean, in given map
    and disk of given centre and radius. Only unmasked pixels by given mask are 
    considered.
    '''
    pixs = getDisk(centre, radius, mask)
    sample = Map[pixs]
    
    N = sample.size
    mean = 1./N * sample.sum()
    var  = np.sqrt( 1./N * ((sample-mean)**2).sum() )
    skew = 1./(N*var**3) * ((sample-mean)**3).sum()
    kur  = 1./(N*var**4) * ((sample-mean)**4).sum() - 3 
    
    return np.array([mean, var, skew, kur])

def chooseSims(radius, nsims=100, plot=True):
    coord = lonlat2colatlon(tcs.COORDCS)
    data, mask = filterMap(MAP, LMAX, radius, mask=True)
    pixs = getDisk(coord, radius, mask)
    TCS = data[pixs].min()
    
    temps = []
    moments = np.zeros((4, nsims+1))
    moments[:,0] = calcStats(coord, radius, MAP.map, MAP.mask)
    count = 1
    while len(temps)<nsims:
        if count%10==0: print(count)
        count +=1
        MAP.genSim(lmax=LMAX)
        fsim = filterMap(MAP, LMAX, radius, sim=True)
        coord = detectCS(fsim, mask)
        pix = hp.ang2pix(MAP.res, coord[0], coord[1])
        if fsim[pix]<TCS*.9:
            temps.append(fsim[pix])
            stats = calcStats(coord, radius, MAP.sim, MAP.mask)
            moments[:, len(temps)] = stats
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(temps, bins=5, normed=False, color='b')
        ax.axvline(x=TCS, color='k', ls='--')
        ax.set_xlabel(r'$T_{cold}$')
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        
        data = moments[:,0]
        sims = moments[:,1:]
        fig = plt.figure()
        c = ('b', 'r', 'y', 'g')
        for i in range(len(MOMENTS)):
            ax = fig.add_subplot(2,2,i+1)
            ax.hist(sims[i,:], bins=10, normed=False, color=c[i])
            ax.axvline(x=data[i], color='k', ls='--')
            ax.set_xlabel(MOMENTS[i])

            ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.tight_layout()
    
    return moments

def angProf(nsims=200, apertures=np.linspace(1, 25, 13), plot=True):
    y = np.zeros((3, 4, apertures.size))
    i = 0
    for R in apertures:
        print('R = ', R)
        m = chooseSims(R, nsims, plot=False)
        data = m[:,0]
        sims = m[:,1:]
        sims_avg = sims.mean(1)
        sims_std = sims.std(1)
        y[:,:, i] = np.vstack((data, sims_avg, sims_std))
        i +=1
    
    if plot:
        fig = plt.figure()
        for i in range(len(MOMENTS)):
            ax = fig.add_subplot(2,2,i+1)
            ax.plot(apertures, y[1,i,:], 'k--')
            ax.plot(apertures, y[0,i,:], 'rx')
            ax.fill_between(apertures, y[1,i,:] - y[2,i,:],
              y[1,i,:] + y[2,i,:], alpha=0.4, facecolor='darkslategray')
            ax.fill_between(apertures, y[1,i,:] - 2*y[2,i,:],
              y[1,i,:] + 2*y[2,i,:], alpha=0.2, facecolor='slategrey')
            ax.set_xlabel(r'Aperture (deg)')
            ax.set_ylabel(MOMENTS[i])
            ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.tight_layout()

def compareFilteredTemp(radius, nsims=100, plot=True):
    '''
    Calculates moments of real map and of nsims in number simulations.
    Moments are based on disk averages. 
    '''
    data, mask = filterMap(MAP, LMAX, radius, mask=True)
    coord = lonlat2colatlon(tcs.COORDCS)
    pixs = getDisk(coord, radius, mask)
    TCS = data[pixs].min()
    
    T = np.zeros(nsims)
    for s in range(nsims):
        if s%10==0: print('sim: ', s)
        MAP.genSim(lmax=LMAX)
        sim = filterMap(MAP, LMAX, radius, sim=True)
        T[s] = sim[mask==1].min()
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(T, bins=20, normed=False, color='b')
        ax.axvline(x=TCS, color='k', ls='--')
        ax.set_xlabel(r'$T_{cold}$')
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    return np.concatenate((np.array([TCS]), T))

def compareFilteredSims(radius, nsims=100, plot=True, bins=10, normed=False):
    '''
    Calculates moments of real map and of nsims in number simulations.
    Moments are based on disk averages. !!! test if CS in sims has mean<0
    '''
    data, mask = filterMap(MAP, LMAX, radius, mask=True)
    coord = lonlat2colatlon(tcs.COORDCS)
    moments = calcStats(coord, radius, data, mask)
    for s in range(nsims):
        MAP.genSim(lmax=LMAX)
        sim = filterMap(MAP, LMAX, radius, sim=True)
        coord = detectCS(sim, mask)
        newmoments = calcStats(coord, radius, sim, mask)
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

def calcArea(nsims=100, thresh=4, apertures=np.linspace(200, 300, 3)):
    apertures = apertures/60
    allAreas = np.zeros((2, apertures.size, nsims+1))
    i = 0
    for R in apertures:   
        print('R = ', R)
        areas = np.zeros((2, nsims+1))
        
        data, mask = filterMap(MAP, LMAX, R, mask=True)
        data = data[mask==1.]
        sigma = data.std()
        areas[0,0] = data[data<-thresh*sigma].size
        areas[1,0] = data[data>thresh*sigma].size
        
        for s in range(nsims):
            if s%50==0: print(s)
            MAP.genSim(lmax=LMAX)
            sim = filterMap(MAP, LMAX, R, sim=True)[mask==1.]
            sigma = sim.std()
            areas[0,s+1] = sim[sim<-thresh*sigma].size
            areas[1,s+1] = sim[sim>thresh*sigma].size
        
        allAreas[:,i,:] = areas
        i +=1
    return allAreas
        







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

def genAngProfiles(nsims=100, apertures=[5], plot=False):
    """
    Plots angluar profiles of all four moments of real data against the 
    average of given number of simulations.
    """
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
'''




































