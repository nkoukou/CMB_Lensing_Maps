'''
Analyses Cold Spot statistics. As of commit 14, the module applies on 
Lensing Maps.

!!!
 - ask about all !!! in other modules
 - What about spots half covered by mask? (maybe ignore discs with more than
   half pixels covered?)
   
 - run top-hat
 
 - class of stats inheriting map?
'''
import time
import numpy as np
import astropy as ap
import healpy as hp
import scipy.ndimage.measurements as snm
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
import LensMapRecon as lmr
from MapFilts import filterMap

# Global constants and functions
MAP = lmr.LensingMap(False, 2048)

DIRFIG = 'Figures/LensSignificance/'
DIRRES = 'CMBL_Maps/results/'

FR = np.linspace(0.5, 15, 30)
FA = np.linspace(0,10,11)

BINS = 10
MOMENTS = ('Mean', 'Variance', 'Skewness', 'Kurtosis')

STR4 = lambda res: str(int(res)).zfill(4)
STR2 = lambda res: str(int(res)).zfill(2)
FNAME = lambda f, R, a, sim, mode: DIRRES+'signif_'+f+'_R'+STR4(R)+'_a'+\
                                   STR2(a)+'_'+STR2(sim)+mode+'.npy'

def lonlat2colatlon(coord):
    '''
    - coord: tuple in form (longitude, latitude)
    Returns tuple in form (colatitude, longitude)
    '''
    lon, lat = coord
    if isinstance(lon, float):
        if lon<0: lon +=360
    else:
        lon[lon<0] +=360
    cb, lon = np.radians(90-lat), np.radians(lon)
    return cb, lon

def colatlon2lonlat(coord):
    '''
    - coord: tuple in form (colatitude, longitude)
    Returns tuple in form (longitude, latitude)
    '''
    cb, lon = coord
    lon, lat = np.rad2deg(lon), 90-np.rad2deg(cb)
    if isinstance(lon, float):
        if lon>180: lon -=360
    else:
        lon[lon>180] -=360
    return lon, lat

###

# Lensing era stats
def selectFilts(sim, scales, alphas, mode, debug=False):
    '''
    Returns given filters for given simulation. Parameters are:
    - sim: integer - simulation number to be considered
    - phi: bool - if True, uses phi map instead of kappa map
    - scales: container (in degrees) - includes the scales of filters to be 
                                       considered
    - alphas: container - includes the alphas of filters to be considered
    - mode: 's' or 'p' - returns significance levels or pixel indices 
                         respectively
    '''
    scales = (60*np.array(scales)).astype(int)
    if MAP.phi: f = 'f'
    else: f = 'k'
    
    if debug: Rmax, amax, sigmax = 0, 0, 0
    data = np.array([])
    for R in scales:
        for a in alphas:
            spots = np.load(FNAME(f, R, a, sim, mode))
            if debug and spots.size!=0:
                if abs(spots).max()>sigmax:
                    Rmax, amax, sigmax = R, a, abs(spots).max()
            data = np.concatenate((data, spots))
    if mode=='p': data = data.astype(int)
    
    if debug:
        return data, (Rmax, amax, sigmax)
    else:
        return data

def pValue(data, sims, metric):
    if metric=='s2n':
        pvalue = sims[sims>data].size/sims.size
    elif metric=='area':
        pvalue = sims[sims>data].size/sims.size
    
    return pvalue

def histSims(scales, alphas, metric, plot=True, debug=False):
    '''
    Returns most extreme values in terms of signal to noise for data and 
    simulations. Parameters are:
    - scales: container - includes the scales of filters to be considered
    - alphas: container - includes the alphas of filters to be considered
    - metric: 'area' or 's2n' - determines if area or signal-to-noise metric is
                                used
    - plot: bool - if True, plots histogram
    '''
    sims = np.zeros(lmr.NSIMS+1)
    
    if metric=='s2n':
        for n in range(lmr.NSIMS+1):
            sig = selectFilts(n, scales, alphas, 's')
            if sig.size==0:
                sims[n] = 0
            else:
                sims[n] = abs(sig).max()
        xlabel = r'Signal to noise ratio for Extreme Spots'
    elif metric=='area':
        for n in range(lmr.NSIMS+1):
            ll, bb, cc = plotFlatExtrema(n, scales, alphas)
            sims[n] = findArea(ll, bb, cc)
        xlabel = r'Number of pixels above threshold of $3\sigma$'
    elif metric=='sigf' or metric=='sigk':
        if metric=='sigf': phi = 0
        else: phi = 1
        
        idxR = np.where( np.isin(FR, scales) )
        idxa = np.where( np.isin(FA, alphas) )
        
        sims = np.load(MAP.dir+'results/extremaTot.npy')[:, phi, idxR, idxa, :]
        sims = abs(sims)
        
        if debug:
            idx = np.where(sims==sims[:,:,:,-1].max())
            filters = [scales[idx[1]], alphas[idx[2]]]
        
        sims = sims.max((0,1,2))
        phi = [r'$\kappa$', r'$\phi$'][phi]
        xlabel = r'Absolute signal for Extreme Spots'
    else:
        raise ValueError('Check the metric')
    
    if plot:
        pvalue = sims[sims>sims[-1]].size/sims.size
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(sims[:-1], bins=BINS, normed=False, histtype='step', lw=1.5)
        ax.axvline(x=sims[-1], color='r', ls='--', label='p = {}'.format(
                   pvalue))
        ax.set_title(xlabel, fontsize=14)
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.legend(loc='upper right', prop={'size':14})
    
    if debug: return sims, filters
    else: return sims

def histPvals(scales, alphas, metric, debug=False):
    '''
    Plots histogram of p-values from histSims() for all simulations and data.
    Parameters are:
    - scales: container - includes the scales of filters to be considered
    - alphas: container - includes the alphas of filters to be considered
    - metric: 'area' or 's2n' - determines if area or signal-to-noise metric is
                                used
    '''
    scales, alphas = np.array(scales), np.array(alphas)
    pvalues = np.zeros((scales.size, alphas.size, lmr.NSIMS+1))
    for s in range(scales.size):
        print(s)
        for a in range(alphas.size):
            sims = histSims([scales[s]], [alphas[a]], metric, plot=False)
            for sim in range(sims.size):
                pvalues[s,a,sim] = sims[sims>sims[sim]].size/sims.size
    
    filters = []
    pvmins = pvalues.min((0,1))
    for i in range(pvmins.size):
        idxs = np.where(pvalues[:,:,i]==pvmins[i])
        scale = scales[np.array(idxs[0])]
        alpha = alphas[np.array(idxs[1])]
        filters.append([scale, alpha])
    
    sims, data = pvmins[:-1], pvmins[-1]
    pv = pvmins[pvmins<=pvmins[-1]].size/pvmins.size
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(sims, bins=BINS, normed=False, histtype='step', lw=1.5)
    ax.axvline(x=data, color='r', ls='--', label='p = {}'.format(pv))
    ax.set_title(r'Most extreme $p$-values')
    ax.legend(loc='upper right', prop={'size':14})
    
    return pvmins, filters

def plotFlatExtrema(sim, scales, alphas, gran=360, plot=False):
    '''
    Returns 2D histogram of the sky with the number of pixels above the 
    threshold, stacking filters in the given range of scales and alphas and 
    binning with given granularity gran in x and y axes (2*gran, gran) as well 
    as plotting if plot=True.
    '''
    pixs = selectFilts(sim, scales, alphas, 'p')
    
    count = np.bincount(pixs)
    count = np.repeat( np.arange(count.size), count )
    cb, lon = hp.pix2ang(MAP.res, count)
    lon, lat = colatlon2lonlat((cb, lon))
    lon, lat = np.around(lon, 3), np.around(lat, 3)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = plt.cm.get_cmap('YlOrBr')
        cmap.set_under('cornflowerblue', 1)
        cc, ll, bb, img = ax.hist2d(lon, lat, bins=(2*gran,gran), vmin=1, 
          range=[[-180, 180], [-90, 90]], normed=False, cmap=cmap)
        ax.invert_xaxis()
        ax.set_xlabel('Longitude (deg)', fontsize=14)
        ax.set_ylabel('Polar angle (deg)', fontsize=14)
        ax.set_title(r'Filtering at $R = ${0}-{1}, $\alpha = ${2}-{3}'.format(
          scales.min(), scales.max(), alphas.min(), alphas.max()), fontsize=16)
        cmap = fig.colorbar(img, ax=ax)
    else:
        cc, ll, bb = np.histogram2d(lon, lat, bins=(2*gran,gran), 
          range=[[-180, 180], [-90, 90]], normed=False)
    return ll, bb, cc

def findArea(ll, bb, cc, coord=False):
    '''
    Finds maximum area metric for given counts cc, with corresponding longitude 
    ll and latitude bb. If coord=True, returns approximate centre of spot too.
    
    !!! does not consider boundary conditions for spots on the 180 degree
        meridian (ll.size, bb.size are 1 more than cc.shape, pixels do not 
        repeat on both ends, np.tile can duplicat cc and ll, averaging ll should
        consider only positive coords)
    '''
    lab, total = snm.label(cc)

    area, n = 0, 0
    for i in range(total):
        if cc[lab==i+1].sum()>area:
            area = cc[lab==i+1].sum()
            n = i+1
    
    if coord:
        bins = np.where(lab==n)
        lon = ll[bins[0]].mean()
        lat = bb[bins[1]].mean()
        coord = lonlat2colatlon((lon, lat))
        coord = (lon, lat)
        area = (area, coord)
    
    return area

def plotMapExtrema(Map, mask, thresh=3, plot=False, savefig=None):
    '''
    Plots map with only unmasked pixels the ones above thresh * std of 
    Map[mask==1] and returns these pixels along with their significance. Plots 
    only if plot=True and saves figures only if savefig is a tuple of scale and
    alpha parameters which the map was filtered with.
    '''
    newmap = np.copy(Map)
    data = newmap[mask==1.]
    sigma = data.std()
    
    newmask = np.zeros(newmap.size, float)
    newmask[newmap<-thresh*sigma] = 1.
    newmask[newmap>thresh*sigma] = 1.
    newmask *=mask
    
    if plot:
        newmap[newmask==0.] = hp.UNSEEN
        newmap = hp.ma(newmap)
        fmt = '%07.3e'
        unt = r'$\kappa$'
        ttl = r'Spots more extreme than {0}$\sigma$'.format(thresh)
        cmap = ListedColormap(np.loadtxt('Figures/cmb_cmap.txt')/255.)
        cmap.set_under('w')
        cmap.set_bad('gray')
        hp.mollview(newmap, title=ttl, 
          format=fmt, cmap=cmap, cbar=True, unit=unt)
    
    if savefig is not None:
        s, a = savefig
        if MAP.phi: strng = '_f'
        else: strng = '_k'
        fname = DIRFIG+STR4(60*s)+'_'+STR2(a)+strng
        plt.savefig(fname)
        plt.close()
    
    pixs = np.where(newmask==1.)[0].astype(int)
    sig = Map[newmask==1.]/sigma
    return pixs, sig

def _plotAllExtrema(sim, scales=FR, alphas=FA, thresh=3, 
                    savefig=None, saveres=False):
    '''
    Saves or returns all extreme spots (in the form of pixels) along with their 
    significance of given map. Parameters:
      - sim: int - indicates simulation number to be used. Real data are 
                   represented by 99.
      - scales, alphas: array - indicate the scale and alpha parameters to be 
                                used by the filter
      - thresh: float - indicate the level of sigmas at which a spoot is 
                        considered extreme 
      - savefig: Nonetype - if savefig is not None, the figures of the extreme 
                            spots at all scales and alphas are saved as well
      - saveres: bool - when True spots along their significance are saved
                        instead of returned 
    '''
    extrema = np.zeros((2, scales.size, alphas.size, 1))
            
    if sim in np.arange(lmr.NSIMS):
        MAP.loadSim(sim)
        is_sim = True
    elif sim==99:
        is_sim = False
    else:
        raise ValueError('Check sim argument')
    
    for i in range(scales.size):
        for j in range(alphas.size):
            print('R, a =', scales[i], ', ', alphas[j])
    
            if savefig is not None:
                savefig = (scales[i], alphas[j])
            
            Map, mask = filterMap(MAP, scales[i], alphas[j], is_sim=is_sim)
            pixs, sig = plotExtrema(Map, mask, thresh, savefig)
            
            diff = extrema.shape[-1] - pixs.size
            if diff>=0:
                pixs = np.pad(pixs, (0,diff), 'constant', constant_values=0)
                sig = np.pad(sig, (0,diff), 'constant', constant_values=0)
            if diff<0:
                extrema = np.pad(extrema, ((0,0),(0,0),(0,0),(0,-diff)), 
                                 'constant', constant_values=0)
            extrema[0,i,j,:] = pixs
            extrema[1,i,j,:] = sig
    
    if saveres:
        np.save(DIRRES+'signifMISS_k_R02700330_a0110_'+STR2(sim), extrema)
    else:
        return extrema

def _saveAllSigma(scales=FR, alphas=FA):
    sigmas = np.zeros((scales.size, alphas.size,lmr.NSIMS+1))
    
    for i in range(scales.size):
        for j in range(alphas.size):
            print('R, a =', scales[i], ', ', alphas[j])
            Map, mask = filterMap(MAP, scales[i], alphas[j], is_sim=False)
            data = Map[mask==1.]
            sigma = data.std()
            sigmas[i,j,-1] = sigma
    
    for s in range(lmr.NSIMS):
        print('\nSIM:', s, '\n')
        for i in range(scales.size):
            for j in range(alphas.size):
                print('R, a =', scales[i], ', ', alphas[j])
                MAP.loadSim(s, False)
                Map, mask = filterMap(MAP, scales[i], alphas[j], is_sim=True)
                data = Map[mask==1.]
                sigma = data.std()
                sigmas[i,j,s] = sigma
        stop = time.time()
        print('{0:.0f} seconds'.format(stop-start))
    np.save('sigmas', sigmas)

def _saveAllExtrema(scales=FR, alphas=FA):
    extrema = np.zeros((2, 2, scales.size, alphas.size,lmr.NSIMS+1))
    start = time.time()
    
    for phi in [0,1]:
        print('phi', phi)            
        MAP = lmr.LensingMap(phi, 2048)
        for sim in range(lmr.NSIMS+1):
            print('sim', sim)
            MAP.loadSim(sim)
            
            stop = time.time()
            print('{0:.0f} seconds'.format(stop-start))
            start = stop
            
            for s in range(scales.size):
                for a in range(alphas.size):
                    print('R:', scales[s])
                    Map, mask = filterMap(MAP, scales[s], alphas[a],is_sim=True)
            
                    data = Map[mask==1.]
                    extrema[0,phi,s,a,sim] = data.max()
                    extrema[1,phi,s,a,sim] = data.min()
            if sim%3==0: 
                np.save(DIRRES+'extremaGauss', extrema)
                print('SAVED')
    return extrema

# Temperature era stats
def detectCS(Map, mask):
    '''
    Returns coordinates of coldest spot on given map. Coldest is defined by 
    lowest filtered temperature.
    '''
    pix = np.where(Map==Map[mask==1].min())[0][0]
    coord = hp.pix2ang(nside=MAP.res, ipix=pix)
    return coord

def detectES(Map, mask):
    '''
    Returns coordinates of most extreme spot on given map. This is the hottest 
    or coldest pixel on the map.
    '''
    pix = np.where(Map==abs(Map)[mask==1].max())[0][0]
    coord = hp.pix2ang(nside=MAP.res, ipix=pix)
    return coord

def getDisk(centre, radius, mask):
    '''
    Returns pixels within the disk of given centre and radius on any map, 
    excluding the boundaries. Only unmasked pixels by given mask are returned.
    '''
    R = np.radians(radius)
    cb, lon = centre
    VEC = hp.ang2vec(cb, lon, lonlat=False)
    pixs = hp.query_disc(MAP.res, vec=VEC, radius=R, inclusive=False)
    pixs = pixs[np.where(mask[pixs]==1.)]
    return pixs

def calcStats(pixs, Map, mask):
    '''
    Calculates the first four moments, starting from the mean, for given map
    pixels.
    '''
    sample = Map[pixs]
    
    N = sample.size
    mean = 1./N * sample.sum()
    var  = np.sqrt( 1./N * ((sample-mean)**2).sum() )
    skew = 1./(N*var**3) * ((sample-mean)**3).sum()
    kur  = 1./(N*var**4) * ((sample-mean)**4).sum() - 3
    
    return np.array([mean, var, skew, kur])

def chooseSims(radius, nsims=99, plot=True):
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

def angProf(nsims=99, apertures=np.linspace(1, 25, 13), plot=True):
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

def compareFilteredTemp(radius, nsims=99, plot=True):
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

def compareFilteredSims(radius, nsims=99, plot=True, bins=10, normed=False):
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

def calcArea(nsims=99, thresh=4, apertures=np.linspace(200, 300, 3)):
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


























