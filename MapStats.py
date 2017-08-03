'''
Analyses Cold Spot statistics.
'''
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
import TempColdSpot as tcs
from MapFilts import filterMap

# Global constants and functions
MOMENTS = ('Mean', 'Variance', 'Skewness', 'Kurtosis') #All moments considered 
                                                       #in the analysis
LMAX = 80

#MAP = tcs.TempMap(256)

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
    R = np.radians(radius)
    cb, lon = centre
    VEC = hp.ang2vec(cb, lon, lonlat=False)
    pixs = hp.query_disc(MAP.res, vec=VEC, radius=R, inclusive=False)
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
    Moments are based on disk averages.
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




























