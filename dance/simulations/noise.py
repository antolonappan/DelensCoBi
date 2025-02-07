import numpy as np
import healpy as hp
import os

from dance.data import PICO

def arc2cl(arc):
    return np.radians(arc/60)**2
def cl2arc(cl):
    return np.rad2deg(np.sqrt(cl))*60

def noise(arr):
    return cl2arc(1/sum(1/arc2cl(arr)))

class Noise:
    def __init__(self,libdir:str,nside:int):
        PICO.directory = os.path.join(libdir)
        self.config = PICO.data
        self.nside = nside
        self.lmax = 3*nside-1
        self.seeds = np.arange(444444,444444+500, dtype=int)

    @property
    def nlev_pol_arr(self):
        return np.radians(np.array(self.config['Baseline'])/60)**2
    
    @property
    def fwhm(self):
        return np.radians(np.array(self.config['FWHM'])/60)
    
    @property
    def __inv_noise__(self):
        return np.nan_to_num(1.0 / self.nlev_pol_arr)

    @property
    def weights(self):
        return self.__inv_noise__ / np.sum(self.__inv_noise__)
    
    
    def depth_p_ilc(self,unit='uk2'):
        depth = 1.0 / np.sum(self.__inv_noise__)
        if unit == 'uk2':
            return depth
        elif unit == 'uk-arcmin':
            return cl2arc(depth)
        else:
            raise ValueError('unit not recognized')
    
    def fwhm_ilc(self,unit='arcmin'):
        if unit == 'rad':
            return np.sum(self.weights * self.fwhm, axis=0)
        elif unit == 'degree':
            return np.degrees(np.sum(self.weights * self.fwhm, axis=0))
        elif unit == 'arcmin':
            return np.degrees(np.sum(self.weights * self.fwhm, axis=0))*60
        else:
            raise ValueError('unit not recognized')
        
    def get_EB(self,i):
        cl = np.ones(self.lmax+1)*self.depth_p_ilc()
        cl[0] = 0
        cl[1] = 0
        np.random.seed(self.seeds[i])
        return hp.synalm([cl,cl,cl,cl*0],self.lmax,new=True)[1:]

    