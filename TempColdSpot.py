"""
Loads Temperature Map of Planck data release (ref. Planck 2015 results XVI).
"""
import os.path
import numpy as np
import scipy.signal as ss
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

# Global constants and functions
DIR = '/media/nikos/00A076B9A076B52E/Users/nkoukou/Desktop/UBC/'
dirfig = lambda ttl: DIR+'drafts/b_draft/figs/'+ttl+'.pdf'
RAWSPEC = np.loadtxt(DIR+'data/aux/cltt.txt')
CMB_CMAP = np.loadtxt(DIR+'data/aux/cmb_cmap.txt')/255.

COORDCS = (-150.6, -57.4) #lon, lat in degrees in Galactic coordinates
BEAM = 5./60 * np.pi/180 #radians

NSIDES = [2**x for x in range(4, 12)]
FWHM = {};
for nside in NSIDES: FWHM[nside] = BEAM * (2048/nside) #radians

LMAX = lambda res: 3*res-1
STR4 = lambda res: str(res).zfill(4)

def fmt(x):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

###

class TempMap(object):
    '''
    Represents the CMB Temperature Map of the Planck 2015 data release.
    '''
    def __init__(self, res=None):
        '''
        Reads the temperature map of given resolution.
        If res=None, the original .fits file is read with resolution 2048. Then,
        all necessary secondary data are also loaded (e.g. mask).
        '''
        self.dir = DIR+'data/maps/n'+STR4(res)
        
        if res is None:
            self.map = hp.read_map(DIR+'data/aux/cmb_temp.fits', field=0, 
                                   verbose=False)
            self.mask = hp.read_map(DIR+'data/aux/cmb_temp.fits', field=1, 
                                    verbose=False)
            
            self.alm = hp.map2alm(self.map)
            self.mlm = hp.map2alm(self.mask)
            
            self.res = hp.get_nside(self.map)
            
        elif res in NSIDES:
            self.res = res
            
            self.map = np.load(DIR+'data/aux/n'+STR4(res)+'_tmap.npy')
            self.alm = np.load(DIR+'data/aux/n'+STR4(res)+'_talm.npy')
            self.mask = np.load(DIR+'data/aux/n'+STR4(res)+'_tmask.npy')
            self.mlm = np.load(DIR+'data/aux/n'+STR4(res)+'_tmalm.npy')
            
        else:
            raise ValueError('Resolution (Nside) must be a power of 2')
        
        self.cb = np.load(DIR+'data/maps/n'+STR4(self.res)+'a_cb.npy')
        self.lon = np.load(DIR+'data/maps/n'+STR4(self.res)+'a_lon.npy')
        self.lmax = LMAX(self.res)
        
        self.sim = self.map
        self.slm = self.alm
    
    # Setters and exporting methods
    def set_res(self, res):
        '''
        Resets object with new resolution.
        '''
        self.__init__(int(res))
    
    def _lowRes(self, res, bd=0.9):
        '''
        Downgrades resolution of map (ref. Planck 2015 results XVI. Section 2).
        '''
        lmax = LMAX(res)
        beam0 = hp.gauss_beam(FWHM[self.res], lmax)
        pixw0 = hp.pixwin(self.res)[:lmax+1]
        beam = hp.gauss_beam(FWHM[res], lmax)
        pixw = hp.pixwin(res)[:lmax+1]
        fl = (beam*pixw)/(beam0*pixw0)
        
        hp.almxfl(self.alm, fl, inplace=True)
        hp.almxfl(self.mlm, fl, inplace=True)
        
        lowmap = hp.alm2map(self.alm, res, verbose=False)
        lowmask = hp.alm2map(self.mlm, res, verbose=False)
        lowmask[lowmask<bd] = 0
        lowmask[lowmask>=bd] = 1
        
        self.map = lowmap
        self.mask = lowmask
    
    def _write(self, resolutions):
        '''
        Produces output data files for given list of resolutions.
        '''
        for res in resolutions:
            if res!=NSIDES[-1]: self._lowRes(res)
            
            np.save(DIR+'data/aux/n'+STR4(res)+'_tmap.npy', self.map)
            np.save(DIR+'data/aux/n'+STR4(res)+'_tmask.npy', self.mask)
            np.save(DIR+'data/aux/n'+STR4(res)+'_talm.npy', self.alm)
            np.save(DIR+'data/aux/n'+STR4(res)+'_tmalm.npy', self.mlm)
    
    # Plotting function
    def plotMap(self, mask=True):
        '''
        Plots map of phi or kappa and includes mask if mask=True.
        '''
        Map = np.copy(self.map)
        unt = r'$T\ (K)$'
        flnm = 'temp_n'+STR4(self.res)
        if mask:
            Map[self.mask==0.] = hp.UNSEEN
            Map = hp.ma(Map)
        
        ticks = np.linspace(0.9*Map.min(), 0.9*Map.max(), 2)
        
        cmap = ListedColormap(CMB_CMAP)
        cmap.set_under('w')
        cmap.set_bad('gray')
        
        hp.mollview(Map, title = '', cmap=cmap, cbar=False)

        fig = plt.gcf()
        ax = plt.gca()
        image = ax.get_images()[0]
        cbar = fig.colorbar(image, ax=ax, orientation='horizontal', pad=0.04,
               fraction=0.05, aspect=26, ticks=[Map.min(),Map.max()])
        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels([fmt(Map.min()), fmt(Map.max())])
        cbar.ax.set_xlabel(unt, fontsize=16, labelpad=-18)
        cbar.ax.tick_params(labelsize=16, pad=10, size=0)
        plt.savefig(dirfig(flnm), bbox_inches='tight')
    

















