"""
Practice on tests of gaussianity.

!!!
optimise noise variance for each pixel?
variance -> skewness, kyrtosis (how to go to these moments in real data?)
simulations (why kurtosis has mean 1?)
"""
import numpy as np
import scipy.stats as ss
import astropy as ap
import healpy as hp
import matplotlib.pylab as plt

from importlib import reload
import LensMapRecon as lmr
reload(lmr)

class NonGauss(lmr.LensingMap):
    """
    Involves several tests of non-Gaussianity for the Planck 2015 Lensing Map.
    """
    def bootKappa(self):
        '''
        Gets relevant converging lens parameters as class attributes.
        '''
        self.lensingMap(phi=False, plot=False)
        self.powerSpectrum(plot=False)
        self.calcNoiseVar()
    
    def calcNoiseVar(self):
        '''
        units???
        '''
        prefac = 1./(4*np.pi) * (2.*np.arange(self.nlkk.size)+1)
        self.noiseVar = np.sum(prefac * self.nlkk)
        return self.noiseVar
    
    def unitLens(self, var):
        return self.map[self.mask==1]/np.sqrt(self.noiseVar + var)
    
    def unitVar(self, var=0.05, step=0.1, negstep=1.5, thresh=1e-5):
        diff = 1.
        switch = False
        
        variances, diffs = [var], [diff]
        counter = 0
        while diff>thresh and counter<300:
            ump = self.unitLens(var)
            uvar = ump.var()
            
            run = abs(uvar-1)
            if run > diff:
                switch = not switch
                step = 0.5*(1.+step)
                negstep = 0.5*(1.+negstep)
            if switch:
                var *=step
            elif not switch:
                var *=negstep
            
            variances.append(var)
            diffs.append(diff)
            diff = run
            counter +=1
            
            if counter%1==0: print(counter, diff, switch)
        return variances, diffs

    def readSim(self, num): #!!! stats with alms
        simklm = hp.read_alm('sims/obs_klms/sim_00{0:02}_klm.fits'.format(num))
        
        simflm = 2./(self.ELL[1:]*(self.ELL[1:]+1.))*simklm[1:]
        simflm = np.concatenate((np.array([0]), flm)) #!!! purify division by 0
        
        simmapUnmasked = hp.alm2map(simflm, nside=self.NSIDE)
        simmap = simmapUnmasked[self.mask==1]
        
        simvar = simmap.var()
        simskew = ss.skew(simmap) 
        simkur = ss.kurtosis(simmap)
        return simvar, simskew, simkur
    
    def compareSim(self):
        var, skew, kur = [], [], []
        for num in range(51,100):
            print(num)
            v, s, k = self.readSim(num)
            var.append(v); skew.append(s); kur.append(k)
        fig = plt.figure()
        axv = fig.add_subplot(311)
        axv.hist(var, bins=5, normed=True, color='r')
        axs = fig.add_subplot(312)
        axs.hist(skew, bins=5, normed=True, color='sandybrown')
        axt = fig.add_subplot(313)
        axt.hist(kur, bins=5, normed=True, color='k')

    def maxVar(self):
        noisevars = np.linspace(0.25, 0.85, 4)
        threshes = np.logspace(-7, -3, 5)
        variances = []
        mask = []
        for v in noisevars:
            self.noiseVar = v
            for t in threshes:
                var, diff = self.unitVar(thresh=t)
                if len(var) < 100:
                    mask.append(1)
                else: mask.append(0)
                
                variances.append(var[-1])
        
        #fig = plt.figure()
        #ax = fig.add_subplot(211)
        #ax.plot(noisevars, variances)
        
        return noisevars, variances, mask
































            
            
