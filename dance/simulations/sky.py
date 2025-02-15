from dance.simulations import CMB, Noise
from dance import mpi
from typing import Dict, Optional, Any, Union, List
import healpy as hp
import numpy as np
import os


class Sky:
    def __init__(
        self,
        libdir:str,
        nside:int,
        nlev_p:float,
        lensed: bool = True,
        model: str = "iso",
        beta: Optional[float]=None,
        Acb: Optional[float]=None,
        verbose: Optional[bool] = True,
        cache: Optional[str] = 'sky',
    ):
        __extname__ = f"_b{beta}" if model == "iso" else f"_Acb{Acb}"
        self.obsdir = os.path.join(libdir,f"obs_N{nside}_m{model}_n{str(nlev_p)}" + __extname__)
        if mpi.rank == 0:
            os.makedirs(self.obsdir,exist_ok=True)
        self.cmb = CMB(libdir,nside,lensed,model,beta,Acb,verbose,cache = True if cache == 'all' else False)
        self.noise = Noise(nside,nlev_p)
        self.nside = nside
        self.lmax = 3*nside-1
        if cache == 'all' or cache == 'sky':
            self.cache = True
        else:
            self.cache = False
        self.lensed = lensed

    def __get_EB__(self,idx:int):
        e,b = self.cmb.get_EB(idx)
        bl = hp.gauss_beam(self.noise.fwhm('rad'),lmax=self.lmax,pol=True).T
        hp.almxfl(e,bl[1],inplace=True)
        hp.almxfl(b,bl[2],inplace=True)
        return [e,b]

    
    def get_EB(self,idx:int,E:bool=True,B:bool=True):
        fname = os.path.join(self.obsdir,f"sims{'' if self.lensed else '_g'}_{idx:04d}.fits")
        if os.path.isfile(fname):
            if E and not B:
                return hp.read_alm(fname,hdu=1)
            elif B and not E:
                return hp.read_alm(fname,hdu=2)
            else:
                return [hp.read_alm(fname,hdu=1),hp.read_alm(fname,hdu=2)]
        else:
            se,sb = self.__get_EB__(idx)
            ne,nb = self.noise.get_EB(idx)
            eb = [se+ne,sb+nb]
            del (se,sb,ne,nb)
            if self.cache:
                hp.write_alm(fname,eb)
            if E and not B:
                return eb[0]
            elif B and not E:
                return eb[1]
            else:
                return eb
    
    def get_T(self,idx:int):
        return np.zeros(hp.Alm.getsize(self.lmax),dtype=np.complex64)
    
    def get_E(self,idx:int):
        return self.get_EB(idx,B=False)

    def get_B(self,idx:int):
        return self.get_EB(idx,E=False)
    

