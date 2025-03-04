from dance import mpi
from dance.simulations import CMB
from typing import Dict, Optional, Any, Union, List
from dance.filtering import WienerFilter
import os
from plancklens import qest, qresp
from plancklens import nhl
from plancklens import utils
import healpy as hp
import matplotlib.pyplot as plt
from dance.utils import slice_alms
import numpy as np
from plancklens.n1 import n1
from dance.utils import get_n0_qe, get_n0_iter



class Reconstruct:
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
        lmax_ivf: Optional[int] = 2048,
        lmax_qlm: Optional[int] = 2048,
        qe_key: Optional[str] = 'p_p',
        verbose: Optional[bool] = True,
        special_case=False,
        delens: Optional[Any] = None
    ):
        wf = WienerFilter(libdir,nside,nlev_p,lensed,model,beta,Acb,lmin_ivf,lmax_ivf,verbose,delens)
        self.wf = wf
        self.lensed = lensed
        self.nlev_p = nlev_p

        __extname__ = f"_b{beta}" if model == "iso" else f"_Acb{Acb}"

        basedir = os.path.join(libdir,f"recon_N{nside}_m{model}_n{nlev_p}_{lmin_ivf}{lmax_ivf}{lmax_qlm}" + __extname__)
        if delens is None:
            qlmdir = os.path.join(basedir,f"qlm{'' if lensed else 'gaus'}")
            qlmcyclicdir = os.path.join(basedir,f"qlm{'' if lensed else 'gaus'}_cyclic")
            qlmpairdir = os.path.join(basedir,f"qlm{'' if lensed else 'gaus'}_pair")
            nOdir = os.path.join(basedir,f"n0{'' if lensed else 'gaus'}")
        else:
            qlmdir = os.path.join(basedir,f"qlm{'' if lensed else 'gaus'}_delens")
            qlmcyclicdir = os.path.join(basedir,f"qlm{'' if lensed else 'gaus'}_delens_cyclic")
            qlmpairdir = os.path.join(basedir,f"qlm{'' if lensed else 'gaus'}_delens_pair")
            nOdir = os.path.join(basedir,f"n0{'' if lensed else 'gaus'}_delens")
        
        qrespdir = os.path.join(basedir,f"qresp{'' if lensed else 'gaus'}")
        n1dir = os.path.join(basedir,f"n1")

        cl_weight = wf.cl_len.copy()
        cl_weight['bb'] *= 0

        self.qlms= qest.library_sepTP(qlmdir, wf.ivfs, wf.ivfs,   wf.cl_len['te'], nside, lmax_qlm=lmax_qlm)
        self.qlms_cyclic = qest.library_sepTP(qlmcyclicdir, wf.ivfs, wf.ivfs_cyclic,   wf.cl_len['te'], nside, lmax_qlm=lmax_qlm)
        self.qlms_pair = qest.library_sepTP(qlmpairdir, wf.ivfs, wf.pairivfs,   wf.cl_len['te'], nside, lmax_qlm=lmax_qlm)
        self.n0 = nhl.nhl_lib_simple(nOdir, wf.ivfs, cl_weight, lmax_qlm)
        self.n1 = n1.library_n1(n1dir,wf.cl_len['tt'],wf.cl_len['te'],wf.cl_len['ee'])


       

        self.qresp = qresp.resp_lib_simple(qrespdir, lmax_ivf, cl_weight, wf.cl_len,
                                 {'t': wf.ivfs.get_ftl(), 'e':wf.ivfs.get_fel(), 'b':wf.ivfs.get_fbl()}, lmax_qlm)
        

        
        assert qe_key in ['p_p','a_p'], "qe_key must be either 'p_p' or 'a_p'"
        self.qe_key = qe_key
        self.source = {'p_p': 'p', 'a_p': 'a'}[qe_key]

        self.lmax_qlm = lmax_qlm

        self.cmb = CMB(libdir,nside,lensed,model,beta,Acb,verbose=False)

        self.norm = self.get_norm()

        self.special_case = special_case

    
    def get_norm(self):
        qresp = self.qresp.get_response(self.qe_key, self.source)
        qnorm = utils.cli(qresp)
        return qnorm
    
    def get_n0(self,i:int,iter:bool=False):
        if not self.special_case:
            return self.n0.get_sim_nhl(i, self.qe_key, self.qe_key)*self.norm**2
        else:
            if iter:
                print('doing iter')
                return get_n0_iter(self.nlev_p)
            else:
                print('doing qe')
                return get_n0_qe(self.nlev_p)
    
    def get_n1(self,i:int):
        if self.source == 'p':
            return self.n1.get_n1(self.qe_key,'p',self.wf.cmb.cl_pp(),self.wf.ivfs.get_ftl(),self.wf.ivfs.get_fel(),self.wf.ivfs.get_fbl(),self.lmax_qlm)*self.norm**2
        else:
            return 0
    
    def get_n0_n1(self,i:int,iter:bool=False):
        return self.get_n0(i,iter) # + self.get_n1(i)
    
    def get_cl_th(self):
        if self.source == 'p':
            return self.cmb.cl_pp()[:self.lmax_qlm + 1]
        else:
            return self.cmb.cl_aa()[:self.lmax_qlm + 1]
    
    def get_wf_fl(self,i:int,iter:bool=False):
        cl_th = self.get_cl_th()
        nl = self.get_n0_n1(i,iter)
        return utils.cli(cl_th + nl)*cl_th
    
    def get_qlm_recon(self,i:int, norm: bool=True, wf: bool=False,which='self'):
        if which == 'cyclic':
            qlm = self.qlms_cyclic.get_sim_qlm(self.qe_key,i)
        elif which == 'pair':
            qlm = self.qlms_pair.get_sim_qlm(self.qe_key,i)
        elif which == 'self':
            qlm = self.qlms.get_sim_qlm(self.qe_key,i)
        else:
            raise ValueError("which must be either 'cyclic', 'pair' or 'self'")
        if wf:
            hp.almxfl(qlm, self.norm, inplace=True)
            return hp.almxfl(qlm, self.get_wf_fl(i))
        if norm:
            return hp.almxfl(qlm,self.norm)
        return qlm
    
    def get_qlm_th(self,i:int, norm: bool=True, wf: bool=False, iter:bool=False):
        if self.source == 'p':
            slm = slice_alms(self.cmb.phi_alm(i),lmax_new=self.lmax_qlm)
        else:
            slm = slice_alms(self.cmb.alpha_alm(i),lmax_new=self.lmax_qlm)
        
        nlm = hp.synalm(self.get_n0_n1(i,iter=iter))
        qlm = slm + nlm
        del (slm,nlm)
        if wf:
            return hp.almxfl(qlm, self.get_wf_fl(i,iter))
        return qlm
    
    def get_qlm(self,i:int, norm: bool=True, wf: bool=False, th: bool=False,iter:bool=False):
        if th:
            print('doing theory')
            return self.get_qlm_th(i,norm,wf,iter)
        assert not self.special_case, "Special case is not implemented for this function"
        return self.get_qlm_recon(i,norm,wf)
    
    def get_qcl(self,i:int, norm: bool=True, wf: bool=False, th: bool=False):
        qlm = self.get_qlm(i,norm,wf,th)
        cl = hp.alm2cl(qlm) - self.get_n0(i) - self.get_n1(i)
        return cl
    
    def plot_qcl(self,i:int, norm: bool=True, wf: bool=False, th: bool=False, which='recon'):
        l = np.arange(self.lmax_qlm + 1)
        plt.figure(figsize=(4,4))
        if which == 'recon':
            qlm = self.get_qlm_recon(i,norm,wf)
            cl = hp.alm2cl(qlm) - self.get_n0(i) #- self.get_n1(i)
            plt.loglog(l**4*cl,label='recon')
        elif which == 'th':
            qlm = self.get_qlm_th(i,norm,wf)
            cl = hp.alm2cl(qlm) - self.get_n0(i) - self.get_n1(i)
            plt.loglog(l**4*cl,label='th')
        elif which == 'cross':
            qlm = self.get_qlm_recon(i,norm,wf)
            qlmth = self.get_qlm_th(i,norm,wf)
            cl = hp.alm2cl(qlm,qlmth)
            plt.loglog(l**4*cl,label='cross')
        elif which == 'all':
            qlm = self.get_qlm_recon(i,norm,wf)
            cl = hp.alm2cl(qlm) - self.get_n0(i) - self.get_n1(i)
            plt.loglog(l**4*cl,label='recon')
            qlm = self.get_qlm_th(i,norm,wf)
            cl = hp.alm2cl(qlm) - self.get_n0(i) - self.get_n1(i)
            plt.loglog(l**4*cl,label='th')
            #qlm = self.get_qlm_recon(i,norm,wf)
            #qlmth = self.get_qlm_th(i,norm,wf)
            #cl = hp.alm2cl(qlm,qlmth)
            #plt.loglog(l**4*cl,label='cross')

        plt.loglog(l**4*self.get_cl_th()[:self.lmax_qlm + 1],label='signal')
        plt.loglog(l**4*self.get_n0(i),label='noise')
        plt.legend()
    
    
