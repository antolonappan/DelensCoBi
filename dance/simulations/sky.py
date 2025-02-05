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
        model: str = "iso",
        beta: Optional[float]=None,
        Acb: Optional[float]=None,
        verbose: Optional[bool] = True,
    ):
        self.obsdir = os.path.join(libdir,f"obs_N{nside}_m{model}" + f"_b{beta}" if model == "iso" else f"_Acb{Acb}")
        if mpi.rank == 0:
            os.makedirs(self.obsdir,exist_ok=True)
        self.cmb = CMB(libdir,nside,model,beta,Acb,verbose)
        self.noise = Noise(libdir,nside)
        self.nside = nside
        self.lmax = 3*nside-1

    def __get_EB__(self,idx:int):
        e,b = self.cmb.get_EB(idx)
        bl = hp.gauss_beam(self.noise.fwhm_ilc('rad'),lmax=self.lmax,pol=True).T
        hp.almxfl(e,bl[1],inplace=True)
        hp.almxfl(b,bl[2],inplace=True)
        return [e,b]

    
    def get_EB(self,idx:int):
        fname = os.path.join(self.obsdir,f"sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            return hp.read_alms(fname,hdu=[1,2])
        else:
            se,sb = self.__get_EB__(idx)
            ne,nb = self.noise.get_EB()
            eb = [se+ne,sb+nb]
            del (se,sb,ne,nb)
            hp.write_alm(fname,eb)
            return eb
    
    def get_T(self,idx:int):
        return np.zeros(hp.Alm.getsize(self.lmax),dtype=np.complex64)
    
    def get_E(self,idx:int):
        fname = os.path.join(self.obsdir,f"sims_{idx:04d}.fits")
        return hp.read_alms(fname,hdu=1)

    def get_B(self,idx:int):
        fname = os.path.join(self.obsdir,f"sims_{idx:04d}.fits")
        return hp.read_alms(fname,hdu=2)
    

