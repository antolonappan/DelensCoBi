from .cmb import CMB
from .noise import Noise
from .sky import Sky
import numpy as np
import healpy as hp
from typing import Dict, Optional, Any, Union, List
from plancklens import utils

class mysims(object):
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
    ):
        self.sky = Sky(libdir,nside,nlev_p,lensed,model,beta,Acb,verbose,cache='sky')

    def hashdict(self):
        return {'sim_lib': 'none'}

    def get_sim_tlm(self,idx):
        return np.zeros(hp.Alm.getsize(self.sky.lmax),dtype=np.complex64)

    def get_sim_elm(self,idx):
        return self.sky.get_E(idx)

    def get_sim_blm(self,idx):
        return self.sky.get_B(idx)
    
    def get_sim_tmap(self,idx):
        tmap = self.get_sim_tlm(idx)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap
    
    def get_sim_pmap(self,idx):
        elm = self.get_sim_elm(idx)
        blm = self.get_sim_blm(idx)
        Q,U = hp.alm2map_spin([elm,blm], self.sky.nside, 2, lmax=self.sky.lmax)
        del elm,blm
        return Q ,U
    
class delensims:

    def __init__(self, delens,theory=False,iter=False):
        self.delens = delens
        self.sky = delens.wf.mysims.sky
        self.flb = utils.cli(delens.wf.cl_len['bb'][:4097]*delens.wf.ivfs.fbl/delens.wf.ivfs.transf['b'])
        self.lmax = self.delens.nside*3 - 1
        self.theory = theory
        self.iter = iter

    def hashdict(self):
        return {'sim_lib': 'none'}

    def get_sim_tlm(self,idx):
        lmax = 3*2048 - 1
        return np.zeros(hp.Alm.getsize(lmax),dtype=np.complex64)

    def get_sim_elm(self,idx):
        return self.sky.get_E(idx)

    def get_sim_blm(self,idx):
        b = self.sky.get_B(idx)
        btemp = self.delens.Btemp(idx,th=self.theory,iter=self.iter)
        b = b - btemp
        del btemp
        return hp.almxfl(b,self.flb)
    
    def get_sim_tmap(self,idx):
        tmap = self.get_sim_tlm(idx)
        tmap = hp.alm2map(tmap,self.delens.nside)
        return tmap
    
    def get_sim_pmap(self,idx):
        elm = self.get_sim_elm(idx)
        blm = self.get_sim_blm(idx)
        Q,U = hp.alm2map_spin([elm,blm], self.delens.nside, 2, lmax=self.lmax)
        del elm,blm
        return Q ,U