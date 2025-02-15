import healpy as hp
from typing import Dict, Optional, Any, Union, List
import lenspyx
from dance.qe import Reconstruct
from dance.filtering import WienerFilter
from dance.utils import slice_alms, bin_cmb_spectrum
import numpy as np
import os
from dance import mpi
import pickle as pl
from tqdm import tqdm
from scipy.signal import savgol_filter

class Delens:
    def __init__(
        self,
        libdir:str,
        nside:int,
        nlev_p:float,
        lensed: bool = True,
        model: str = "iso",
        beta: Optional[float]=None,
        Acb: Optional[float]=None,
        lmin_ivf: Optional[int] = 2,
        lmax_ivf: Optional[int] = 3000,
        lmax_qlm: Optional[int] = 3000,
        qe_key: Optional[str] = 'p_p',
        lmin_delens: Optional[int] = 2,
        lmax_delens: Optional[int] = 3000,
        verbose: Optional[bool] = True,
    ):
        self.basedir = os.path.join(libdir,f"delens_N{nside}_m{model}_n{nlev_p}_ivf{lmin_ivf}_{lmax_ivf}_qlm{lmax_qlm}_delens{lmin_delens}_{lmax_delens}")
        if mpi.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
        self.model = model
        self.beta = beta
        self.recon = Reconstruct(libdir,nside,nlev_p,lensed,model,beta,Acb,lmin_ivf,lmax_ivf,lmax_qlm,qe_key,verbose)
        self.wf = WienerFilter(libdir,nside,nlev_p,lensed,model,beta,Acb,lmin_ivf,lmax_ivf,verbose)
        self.lmin_delens = lmin_delens
        self.lmax_delens = lmax_delens
        self.geom_info = ('healpix', {'nside':nside})
        self.verbose = verbose
        self.lensed = lensed
        fl = np.ones(lmax_delens + 1)
        fl[:lmin_delens] = 0
        self.fl = fl


    def grad_phi_alm(self, idx: int, th: bool = False):
        qlm = self.recon.get_qlm(idx,wf=True,th=th)
        hp.almxfl(qlm,self.fl,inplace=True)
        lmax = hp.Alm.getlmax(len(qlm))
        return -hp.almxfl(qlm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)

    def delens(self,i,recon=True):
        fname = os.path.join(self.basedir,f"delens_{'r'if recon else 'g'}{'' if self.lensed else 'gaus' }_{i:04d}.fits")
        if os.path.isfile(fname):
            return hp.read_alm(fname,hdu=1), hp.read_alm(fname,hdu=2)
        else:
            dlm = self.grad_phi_alm(i,th=recon)
            e = self.wf.get_wf_E(i)
            b = self.wf.get_wf_B(i)
            
            Qdelen, Udelen = lenspyx.alm2lenmap_spin([e,b], dlm, 2, geometry=self.geom_info, verbose=int(self.verbose))

            eb = hp.map2alm_spin([Qdelen,Udelen],2)
            hp.write_alm(fname,eb)
            return eb
        
    def _delens_cl_(self,i,recon=True):
        fname = os.path.join(self.basedir,f"delenscl_{'r'if recon else 'g'}{'' if self.lensed else 'gaus' }_{i:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            e,b = self.delens(i,recon)
            cl = hp.alm2cl(e,b)
            pl.dump(cl,open(fname,'wb'))
            return cl
    
    def _delens_cl_transfer_(self,i,recon=True,transfer=False):
        if transfer:
            t = self.wf.get_transfer()
            return self._delens_cl_(i,recon)[:len(t)]/t
        return self._delens_cl_(i,recon)
    
    def delens_cl(self,i,recon=True,transfer=False,bw=None):
        cl = self._delens_cl_transfer_(i,recon,transfer)
        if bw is not None:
            return bin_cmb_spectrum(cl,bw)
        return cl

        
        
