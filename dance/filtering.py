import numpy as np
import healpy as hp

from dance.simulations import CMB, mysims, Noise
from dance import mpi
from typing import Dict, Optional, Any, Union, List
import os
from plancklens import utils
from plancklens.filt import filt_simple
import pickle as pl
from tqdm import tqdm
from scipy.signal import savgol_filter
from dance.utils import bin_cmb_spectrum
from dance.simulations import delensims

class WienerFilter:
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
        lmax_ivf: Optional[int] = 4096,
        verbose: Optional[bool] = True,
        delens: Optional[Any] = None
    ):
        __extname__ = f"_b{beta}" if model == "iso" else f"_Acb{Acb}"
        self.basedir = os.path.join(libdir,f"filt_N{nside}_m{model}_n{str(nlev_p)}_{lmin_ivf}{lmax_ivf}" + __extname__)
        if mpi.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
        self.model = model
        self.beta = beta
        sims = mysims(libdir,nside,nlev_p,lensed,model,beta,Acb,verbose)
        self.mysims = sims
        self.cmb = CMB(libdir,nside,lensed,model,beta,Acb,verbose)
        noise = Noise(nside,nlev_p)

        self.lmin_ivf = lmin_ivf
        self.lmax_ivf = lmax_ivf
        self.nlev_p = nlev_p
        self.nlev_t = self.nlev_p * np.sqrt(2)

        cl_len = self.cmb.get_lensed_spectra(dl=False)
        self.cl_len = cl_len
         
        transf = hp.gauss_beam(noise.fwhm('arcmin') / 60. / 180. * np.pi, lmax=lmax_ivf) * hp.pixwin(nside)[:lmax_ivf + 1]

        ftl = utils.cli(cl_len['tt'][:lmax_ivf + 1] + (self.nlev_t / 60. / 180. * np.pi / transf) ** 2)
        fel = utils.cli(cl_len['ee'][:lmax_ivf + 1] + (self.nlev_p / 60. / 180. * np.pi / transf) ** 2)
        fbl = utils.cli(cl_len['bb'][:lmax_ivf + 1] + (self.nlev_p / 60. / 180. * np.pi / transf) ** 2)
        ftl[:lmin_ivf] *= 0.
        fel[:lmin_ivf] *= 0.
        fbl[:lmin_ivf] *= 0.

        

        if delens is None:
            self.ivfs = filt_simple.library_fullsky_sepTP(self.basedir, sims, nside, transf, cl_len, ftl, fel, fbl,)
        else:
            print('Delens Filtering')
            self.mysims = delensims(delens)
            self.ivfs = filt_simple.library_fullsky_sepTP(self.basedir, delensims(delens), nside, transf, cl_len, ftl, fel, fbl)

    def get_wf_E(self,i):
        return self.ivfs.get_sim_emliklm(i)
        
    def get_wf_B(self,i):
        return self.ivfs.get_sim_bmliklm(i)
    
    def _get_wf_EB_(self,i):
        fname = os.path.join(self.basedir, f"eb_cl_{i:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))
        else:
            eb = hp.alm2cl(self.ivfs.get_sim_emliklm(i), self.ivfs.get_sim_bmliklm(i))
            pl.dump(eb, open(fname, "wb"))
            return eb
    
    # def get_wf_EB_trans(self,i,transfer=False):
    #     if transfer:
    #         return self._get_wf_EB_(i) / self.get_transfer()
    #     return self._get_wf_EB_(i)
    
    # def get_wf_EB(self,i,transfer=False,bw=None):
    #     cl = self.get_wf_EB_trans(i,transfer)
    #     if bw is not None:
    #         return bin_cmb_spectrum(cl,bw)
    #     return cl
        
    # def get_transfer(self):
    #     assert self.model == 'iso', 'Only isotropic model is supported'
    #     fname = os.path.join(self.basedir, f"transfer.pkl")
    #     if os.path.isfile(fname):
    #         return pl.load(open(fname, "rb"))
    #     else:
    #         eb = []
    #         for i in tqdm(range(300),desc='Computing Transfer Functions'):
    #             eb.append(self.get_wf_EB(i))
    #         eb = np.array(eb)
    #         eb = np.mean(eb,axis=0)
    #         EB = self.cmb.get_cb_lensed_spectra(self.beta,dl=False)['eb']
    #         EB = EB[:len(eb)]
    #         transfer = eb/EB
    #         transfer[np.isnan(transfer)] = 0
    #         stransfer = savgol_filter(transfer[2:], window_length=300, polyorder=3)
    #         stransfer=np.concatenate(([0, 0], stransfer))
    #         pl.dump(stransfer, open(fname, "wb"))
    #         return stransfer



