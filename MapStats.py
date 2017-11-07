'''
Analyses Cold Spot statistics. As of commit 14, the module applies on 
Lensing Maps.

!!!
 - ask about all !!! in other modules
 - What about spots half covered by mask? (maybe ignore discs with more than
   half pixels covered? should not do that assuming that highest signal is 
   correct metric after filtering)
 - class of stats inheriting map?
#!!!!!!!! Check indices 0 and -1 for data vs simulations, divide by 100 or 101 for pvalue??, why pvalue in pvalueHist is so high???, xCorr
'''
import os.path
import time
import numpy as np
import astropy as ap
import healpy as hp
import matplotlib as mat
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
import scipy.ndimage.measurements as snm
import LensMapRecon as lmr
import TempColdSpot as tcs
from MapFilts import filterMap

# Global constants and functions
MAP = lmr.LensingMap(phi=False, conserv=False, res=256)

font = {'size'   : 12}
lines = {'lw'   : 1.}
mat.rc('font', **font)
mat.rc('lines', **lines)

DIR = '/media/nikos/00A076B9A076B52E/Users/nkoukou/Desktop/UBC/'

NSIDES = [2**x for x in range(4, 12)]
NSIMS = 100

FRTEST = np.array([25, 50, 100, 1000]) #10 gives problems with mask
FRN = np.linspace(30, 900, 30)
FR = np.concatenate((FRTEST, FRN))
FA = np.linspace(0,10,11)

BINS = 10
MOMENTS = ('Mean', 'Variance', 'Skewness', 'Kurtosis')
QUANTS = (r'$\kappa$', r'$\phi$')

STR2 = lambda res: str(int(res)).zfill(2)
STR4 = lambda res: str(int(res)).zfill(4)
FNAME = lambda f, R, a, sim, mode: DIRRES+'signif_'+f+'_R'+STR4(R)+'_a'+\
                                   STR2(a)+'_'+STR2(sim)+mode+'.npy'
CSTR = lambda conserv: 'C' if conserv else ''
FK = lambda phi: 'f' if phi else 'k'
###

# Tests
def moments(conserv=False, plot=True, res=None):
    '''
    '''
    MAP.set_res(res)
    MAP.set_conserv(conserv)
    
    cstr = CSTR(conserv)
    datadir = DIR+'results/n'+STR4(MAP.res)+'_moments'+cstr+'.npy'
    if os.path.isfile(datadir):
        data = np.load(datadir)
    else:
        data = np.zeros((2,3,NSIMS+1))
        for p in [0,1]:
            MAP.set_phi(p)
            for s in range(NSIMS+1):
                print(p,s)
                MAP.loadSim(s)
                mean, var, skew, kur = calcStats(MAP.sim, MAP.mask, pixs=None)
                data[p,0,s] = var; data[p,1,s] = skew; data[p,2,s] = kur
        np.save(datadir, data)
    
    if plot:
        fig, axarr = plt.subplots(2, 3)
        for p in [0,1]:
            for i in range(3):
                ax = axarr[p,i]
                ax.hist(data[p,i,:-1], bins=BINS, normed=False, histtype='step', 
                        color='b')
                ax.axvline(x=data[p,i,-1], color='r', ls='--')
                
                if p==1: ax.set_xlabel(MOMENTS[i+1], fontsize=12, labelpad=15)
                if i==0: ax.set_ylabel(QUANTS[p], fontsize=12)
                ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    
    return data

def csaverages(conserv=False, plot=True, res=None):
    '''
    '''
    MAP.set_res(res)
    MAP.set_conserv(conserv)
    
    radii = np.linspace(0.2, 40, 100)
    cstr = CSTR(conserv)
    datadir = DIR+'results/n'+STR4(MAP.res)+'_csAnuli'+cstr+'.npy'
    if os.path.isfile(datadir):
        anuli = np.load(datadir)
    else:
        tcoord = lonlat2colatlon(tcs.COORDCS)
        
        tmap = tcs.TempMap(MAP.res)
        tmap, tmask = tmap.map, tmap.mask
        tsigma = tmap[tmask==1.].std()
        
        MAP.set_phi(False)
        kmap, kmask = MAP.map, MAP.mask
        ksigma = kmap[kmask==1.].std()
        MAP.set_phi(True)
        fmap, fmask = MAP.map, MAP.mask
        fsigma = fmap[fmask==1.].std()
        
        sizes = np.zeros((3, radii.size))
        averages = np.zeros((3, radii.size))
        i=0
        for radius in radii:
            if i%50==0: print(i)
            pixs = getDisk(tcoord, radius, tmask)
            sizes[0,i] = pixs.size
            averages[0,i] = tmap[pixs].mean()
            pixs = getDisk(tcoord, radius, kmask)
            sizes[1,i] = pixs.size
            averages[1,i] = kmap[pixs].mean()
            pixs = getDisk(tcoord, radius, fmask)
            sizes[2,i] = pixs.size
            averages[2,i] = fmap[pixs].mean()
            i +=1
        
        anuli = np.zeros((3, radii.size-1))
        for i in range(averages.shape[1]-1):
            anuli[:,i] = sizes[:,i+1]/sizes[:,i] * averages[:,i+1] - \
                           sizes[:,i]/(sizes[:,i+1]-sizes[:,i]) * averages[:,i]
        anuli[0] /= tsigma; anuli[1] /= ksigma; anuli[2] /= fsigma
        
        np.save(DIR+'results/n'+STR4(MAP.res)+'_tsigma'+cstr, tsigma)
        np.save(DIR+'results/n'+STR4(MAP.res)+'_ksigma'+cstr, ksigma)
        np.save(DIR+'results/n'+STR4(MAP.res)+'_fsigma'+cstr, fsigma)
        np.save(DIR+'results/n'+STR4(MAP.res)+'_spotSizes', sizes)
        np.save(DIR+'results/n'+STR4(MAP.res)+'_csAverages'+cstr, averages)
        np.save(datadir, anuli)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(radii[1:], anuli[0], 'r-', label=r'$T$')
        ax.plot(radii[1:], anuli[1], 'b:', label=QUANTS[0])
        ax.plot(radii[1:], anuli[2], 'g--', label=QUANTS[1])
        
        ax.set_xlabel(r'Radius (deg)')
        ax.legend(loc='upper right', prop={'size':14})
        
    return anuli

def xCorrelate(radius, phi=False, conserv=False, plot=True, res=None):
    '''
    Correlates temperature Cold Spot with lensing map.
    '''
    tick = time.time()
    
    MAP.set_phi(phi)
    MAP.set_res(res)
    MAP.set_conserv(conserv)
    
    fk = FK(phi)
    cstr = CSTR(conserv)
    datadir = (DIR+'results/n'+STR4(MAP.res)+fk+'_xcorr_r'+STR2(radius)+cstr+
               '.npy')
    print(datadir)
    if os.path.isfile(datadir):
        xcorr = np.load(datadir)
    else:
        tmap = tcs.TempMap(MAP.res)
        lenslons = np.arange(0,360,2*radius)
        lenslats = np.arange(0,180,2*radius)
    
        xcorr = np.zeros((4, lenslons.size, lenslats.size))
        print('START: {:.1f}'.format(time.time() - tick)); tick = time.time()
        for lon in range(lenslons.size):
            for lat in range(lenslats.size):
                
                rotate(tmap, lenslons[lon], lenslats[lat])
                
                nmp = filterMap(tmap, 300, 2, mask=False)
                nmk = np.ones(nmp.size)
                tcoord = detectES(nmp, nmk, 'c')
                tpixs = getDisk(tcoord, radius)
                size = np.where(MAP.mask[tpixs]==1.)[0].size
                corrval = np.sum(tmap.sim[tpixs]*MAP.map[tpixs])
                
                print('ROUND: {:.1f}'.format(time.time() - tick)); tick = time.time()
                print('LON, LAT:', lenslons[lon], lenslats[lat])
                plon, plat = colatlon2lonlat(tcoord)
                print('CENTRE: ({0:.2f}, {1:.2f})'.format(plon, plat))
                print('SIZE / PIXS: ({0} / {1})'.format(size, tpixs.size))
                print()
                
                xcorr[0, lon, lat] = corrval
                xcorr[1, lon, lat] = size
                xcorr[2, lon, lat] = tcoord[0]
                xcorr[3, lon, lat] = tcoord[1]
                
        np.save(datadir, xcorr)
    
    dirs = xcorr[0][xcorr[1]>0.9*xcorr[1].max()]
    dircs = dirs[0]
    ratio = dirs.size/xcorr[0].size
    pvalue = dirs[dirs>dircs].size/(dirs.size+1)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(dirs[1:], bins=BINS, normed=False, histtype='step', lw=1.5)
        ax.axvline(x=dircs, color='r', ls='--', label='p = {0:.2f}'.format(
                   pvalue))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.set_xlabel(r'Correlation value')
        ax.legend(loc='upper right', prop={'size':14})
    
    return xcorr, pvalue, ratio

def histSims(scales, alphas, metric, phi=False, conserv=False, plot=True, 
             res=None):
    '''
    !!!
    Returns most extreme values in terms of signal to noise for data and 
    simulations. Parameters are:
    - scales: container - includes the scales of filters to be considered
    - alphas: container - includes the alphas of filters to be considered
    - metric: 'area' or 's2n' - determines if area or signal-to-noise metric is
                                used
    - plot: bool - if True, plots histogram
    '''
    MAP.set_phi(phi)
    MAP.set_res(res)
    MAP.set_conserv(conserv)
    
    scales = np.array(scales)
    alphas = np.array(alphas)
    
    fk = FK(phi)
    cstr = CSTR(conserv)
    datadir = DIR+'results/n'+STR4(MAP.res)+fk+'_univs'+cstr+'.npy'
    if os.path.isfile(datadir):
        univs = np.load(datadir)
    else:
        sims = lmr.NSIMS+1
        univs = np.zeros((2, FR.size, FA.size, sims))
        moments = np.zeros((len(MOMENTS),FR.size, FA.size, sims))
        
        for n in range(sims):
            MAP.loadSim(n)
            for s in range(FR.size):
                for a in range(FA.size):
                    print('R, a = ', FR[s], FA[a], '||', n)
                    nmp, nmk = filterMap(MAP, FR[s], FA[a])
                    univs[0,s,a,n] = nmp[nmk==1].min()
                    univs[1,s,a,n] = nmp[nmk==1].max()
                    moments[:,s,a,n] = calcStats(nmp, nmk, None)
        
        np.save(datadir, univs)
        for i in range(len(MOMENTS)):
            datadir= DIR+'results/n'+STR4(MAP.res)+fk+'_'+MOMENTS[i]+cstr+'.npy'
            np.save(datadir, moments[i])
    
    fr = np.array(np.where(np.in1d(FR, scales))[0])
    fa = np.array(np.where(np.in1d(FA, alphas))[0])
    
    if metric=='abssig':
        univs = univs[:,fr,:,:][:,:,fa,:]
        sims = abs(univs).max((0,1,2))
        pvalue = sims[sims>sims[-1]].size/sims.size
    else:
        raise Exception('Not yet implemented')
    
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(sims[:-1], bins=BINS, normed=False, histtype='step', lw=1.5)
        ax.axvline(x=sims[-1], color='r', ls='--', label='p = {0:.2f}'.format(
                   pvalue))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.legend(loc='upper right', prop={'size':14})
    
    return pvalue, sims

def plotPvalues(scales, alphas, metric, phi=False, conserv=False, plot=True, 
               res=None):
    '''
    '''
    plts = plta = False
    if len(scales) > 1:
        scales = np.array(scales); plts = True
    if len(alphas) > 1:
        alphas = np.array(alphas); plta = True
    if plts==plta==False:
        raise ValueError('Neither scales nor alphas are arrays')
    
    fk = FK(phi)
    cstr = CSTR(conserv)
    datadir = DIR+'results/n'+STR4(MAP.res)+fk+'_pvalues_'+metric+cstr+'.npy'
    if os.path.isfile(datadir):
        pvalues = np.load(datadir)
    else:
        pvalues = np.zeros((FR.size, FA.size))
        for s in range(FR.size):
            for a in range(FA.size):
                pvalues[s,a], sims = histSims([FR[s]], [FA[a]], metric, 
                                 phi=False, conserv=False, plot=False, res=None)
        np.save(datadir, pvalues)
    
    if plot:
        fr = np.array(np.where(np.in1d(FR, scales))[0])
        fa = np.array(np.where(np.in1d(FA, alphas))[0])
        yaxis = pvalues[fr,:][:,fa]
        if plts!=plta:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if plts:
                xaxis = scales; p = alphas[0]
                xlabel = r'$R (^\prime)$'
            if plta:
                xaxis = alphas; p = scales[0]
                xlabel = r'$\alpha$'
            ax.plot(xaxis, np.squeeze(yaxis), 'bx:')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'$p$-value')
        
        if plts and plta:
            pass

def histPvalues(scales, alphas, metric, phi=False, conserv=False, plot=True, 
                res=None):
    '''
    !!!
    Plots histogram of p-values from histSims() for all simulations and data.
    Parameters are:
    - scales: container - includes the scales of filters to be considered
    - alphas: container - includes the alphas of filters to be considered
    - metric: 'area' or 's2n' - determines if area or signal-to-noise metric is
                                used
    '''
    scales, alphas = np.array(scales), np.array(alphas)
    pvalues = np.zeros((scales.size, alphas.size, lmr.NSIMS+1))
    
    cstr = CSTR(conserv)
    fk = FK(phi)
    datadir = DIR+'results/n'+STR4(MAP.res)+fk+'_univs'+cstr+'.npy'
    sims = np.load(datadir)
    fr = np.array(np.where(np.in1d(FR, scales))[0])
    fa = np.array(np.where(np.in1d(FA, alphas))[0])
    sims = sims[:,fr,:,:][:,:,fa,:]
    
    if metric=='abssig':
        sims = abs(sims).max(0)
        for s in range(scales.size):
            for a in range(alphas.size):
                univ = sims[s,a,:]
                for sim in range(lmr.NSIMS+1):
                    pvalues[s,a,sim] = univ[univ>univ[sim]].size/univ.size
    else:
        raise Exception('Not yet implemented')
    
    filters = []
    pvmins = pvalues.min((0,1))
    for i in range(pvmins.size):
        idxs = np.where(pvalues[:,:,i]==pvmins[i])
        scale = scales[np.array(idxs[0])]
        alpha = alphas[np.array(idxs[1])]
        filters.append([scale, alpha])
    
    sims, data = pvmins[:-1], pvmins[-1]
    pv = pvmins[pvmins<=pvmins[-1]].size/pvmins.size
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(sims, bins=BINS, normed=False, histtype='step', lw=1.5)
        ax.axvline(x=data, color='r', ls='--', label='p = {0:.2f}'.format(pv))
        ax.set_title(r'Most extreme $p$-values')
        ax.legend(loc='upper right', prop={'size':14})
    
    return pvmins, filters            

# Basic pixel selection functions
def lonlat2colatlon(coord):
    '''
    - coord: tuple in form (longitude, latitude)
    Returns tuple in form (colatitude, longitude)
    '''
    lon, lat = coord
    if isinstance(lon, float) or isinstance(lon, int):
        if lon<0: lon +=360
    else:
        lon[lon<0] +=360
    cb, lon = np.radians((90-lat)%180), np.radians(lon)
    return cb, lon

def colatlon2lonlat(coord):
    '''
    - coord: tuple in form (colatitude, longitude)
    Returns tuple in form (longitude, latitude)
    '''
    cb, lon = coord
    lon, lat = np.rad2deg(lon), 90-np.rad2deg(cb)
    if isinstance(lon, float) or isinstance(lon, int):
        if lon>180: lon -=360
    else:
        lon[lon>180] -=360
    return lon, lat

def getDisk(centre, radius, mask=None, edge=False):
    '''
    Returns pixels within the disk of given centre (cb, lon) and radius (deg) on
    any map, excluding the boundaries. Only unmasked pixels by given mask are 
    returned.
    '''
    R = np.radians(radius)
    cb, lon = centre
    VEC = hp.ang2vec(cb, lon, lonlat=False)
    pixs = hp.query_disc(MAP.res, vec=VEC, radius=R, inclusive=edge)
    if mask is not None: pixs = pixs[np.where(mask[pixs]==1.)]
    return pixs

def rotate(tmap, lon, lat):
    '''
    Rotates given map by given longitude and latitude, also plots the stages of 
    the rotation if plot=True. Stores rotated map in the sim and slm attributes 
    of tmap.
    '''
    t, p = hp.pix2ang(tmap.res, np.arange(tmap.map.size))
    coord = tcs.COORDCS
    
    rot = hp.Rotator(rot=(-coord[0],0,0))
    tn, pn = rot(t,p)
    pixs = hp.ang2pix(tmap.res, tn, pn)
    tmap.sim = tmap.map[pixs]
    
    rot = hp.Rotator(rot=(0,lat,0))
    tn, pn = rot(t,p)
    pixs = hp.ang2pix(tmap.res, tn, pn)
    tmap.sim = tmap.sim[pixs]
    
    rot = hp.Rotator(rot=(coord[0]+lon,0,0))
    tn, pn = rot(t,p)
    pixs = hp.ang2pix(tmap.res, tn, pn)
    tmap.sim = tmap.sim[pixs]
    tmap.slm = hp.map2alm(tmap.sim)

def detectES(Map, mask, hc):
    '''
    Returns coordinates of most extreme spot on given map. Parameter hc can be:
    'h' for hot, 'c' for cold, 'hc' for hot and cold and '' for hot or cold.
    '''
    pixmax = np.where(Map==Map[mask==1.].max())[0][0]
    pixmin = np.where(Map==Map[mask==1.].min())[0][0]
    if hc=='h': pix = pixmax
    elif hc=='c': pix = pixmin
    elif hc=='hc': pix = (pixmin, pixmax)
    elif hc=='': pix = float(np.where(abs(pixmin)>abs(pixmax), pixmin, pixmax))
    else:
        raise ValueError('Check parameter hc')
    coord = hp.pix2ang(nside=MAP.res, ipix=pix)
    return coord

def calcStats(Map, mask, pixs):
    '''
    Calculates the first four moments, starting from the mean, for given map
    pixels.
    '''
    if pixs is None:
        sample = Map
        if mask is not None: sample = Map[mask==1.]
    else:
        sample = Map[pixs]
        if mask is not None: sample = sample[mask[pixs]==1.]
    
    N = sample.size
    mean = 1./N * sample.sum()
    var  = np.sqrt( 1./N * ((sample-mean)**2).sum() )
    skew = 1./(N*var**3) * ((sample-mean)**3).sum()
    kur  = 1./(N*var**4) * ((sample-mean)**4).sum() - 3
    
    return np.array([mean, var, skew, kur])

# Lensing era stats
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


























