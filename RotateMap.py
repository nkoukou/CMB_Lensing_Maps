# coding: utf-8
import healpy as hp
import numpy as np
import matplotlib.pylab as plt
import TempColdSpot as tcs
from MapFilts import filterMap
from MapStats import detectES
from MapStats import colatlon2lonlat

tmap = tcs.TempMap(256)
nmp, nmk= filterMap(tmap, 300, 2)
tmap.map = nmp
cslon, cslat = tcs.COORDCS

def rotate(lon, lat, plot):
    t, p = hp.pix2ang(tmap.res, np.arange(tmap.map.size))
    coord1 = colatlon2lonlat(detectES(tmap.map, tmap.mask, 'c'))
    b = tmap.map
    if plot: tmap.plotMap()
    
    rot = hp.Rotator(rot=(-coord1[0],0,0))
    tn, pn = rot(t,p)
    pixs = hp.ang2pix(tmap.res, tn, pn)
    tmap.map = tmap.map[pixs]
    coord2 = colatlon2lonlat(detectES(tmap.map, tmap.mask, 'c'))
    if plot:tmap.plotMap()
    
    rot = hp.Rotator(rot=(0,lat,0))
    tn, pn = rot(t,p)
    pixs = hp.ang2pix(tmap.res, tn, pn)
    tmap.map = tmap.map[pixs]
    coord3 = colatlon2lonlat(detectES(tmap.map, tmap.mask, 'c'))
    if plot:tmap.plotMap()
    
    rot = hp.Rotator(rot=(coord1[0]+lon,0,0))
    tn, pn = rot(t,p)
    pixs = hp.ang2pix(tmap.res, tn, pn)
    tmap.map = tmap.map[pixs]
    coord4 = colatlon2lonlat(detectES(tmap.map, tmap.mask, 'c'))
    if plot:tmap.plotMap()
    tmap.map = b
    return coord1, coord2, coord3, coord4

def optimise():
    bs = np.linspace(1,45,45)
    cs = 3/5*bs
    lons, lats = np.zeros(bs.size), np.zeros(bs.size)
    for i in range(bs.size):
        print('STEP:', i)
        eps, diff, step = 0.2, 1, 1/(i+1)
        while (abs(diff)>eps and step>1e-10):
            lons[i], lats[i] = rotate(0,bs[i],cs[i],False)
            cs[i] += step
            if abs(lons[i]-cslon) > abs(diff): step = -step/2
            diff = lons[i]-cslon
            print(lons[i], diff, step)
    return bs, cs, lons, lats
