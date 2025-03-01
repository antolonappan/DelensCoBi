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
from plancklens import utils

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
        lmin_ivf: Optional[int] = 150,
        lmax_ivf: Optional[int] = 2048,
        lmax_qlm: Optional[int] = 2048,
        qe_key: Optional[str] = 'p_p',
        lmin_delens: Optional[int] = 8,
        verbose: Optional[bool] = True,
        special_case=False,
    ):  
        self.libdir = libdir
        self.basedir = os.path.join(libdir,f"delens_N{nside}_m{model}_n{nlev_p}_ivf{lmin_ivf}_{lmax_ivf}_qlm{lmax_qlm}_delens{lmin_delens}")
        if mpi.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
        self.nside = nside
        self.model = model
        self.beta = beta
        self.lmin_ivf = lmin_ivf
        self.lmax_ivf = lmax_ivf
        self.lmax_qlm = lmax_qlm
        self.lmin_delens = lmin_delens
        self.lensed = lensed
        self.nlev_p = nlev_p
        self.recon = Reconstruct(libdir,nside,nlev_p,lensed,model,beta,Acb,lmin_ivf,lmax_ivf,lmax_qlm,qe_key,verbose,special_case=special_case)
        self.wf = WienerFilter(libdir,nside,nlev_p,lensed,model,beta,Acb,lmin_ivf,lmax_ivf,verbose)
        self.lmin_delens = lmin_delens
        self.geom_info = ('healpix', {'nside':nside})
        self.verbose = verbose
        self.lensed = lensed
        fl = np.ones(lmax_qlm+ 1)
        fl[:lmin_delens] = 0
        self.fl = fl
        self.special_case = special_case


    def grad_phi_alm(self, idx: int, th: bool = False,iter: bool = False):
        qlm = self.recon.get_qlm(idx,wf=True,th=th,iter=iter)
        hp.almxfl(qlm,self.fl,inplace=True)
        lmax = hp.Alm.getlmax(len(qlm))
        return hp.almxfl(qlm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)), None, False)

    def Btemp(self,i,th=True,iter=False):
        fname = os.path.join(self.basedir,f"bmode_temp_{'g'if th else 'r'}{'' if self.lensed else 'gaus' }{'_i' if iter else ''}{'_s' if self.special_case else ''}_{i:04d}.fits")
        if os.path.isfile(fname):
            return hp.read_alm(fname,hdu=1)
        else:
            dlm = self.grad_phi_alm(i,th,iter)
            e = self.wf.get_wf_E(i)
            Qdelen, Udelen = lenspyx.alm2lenmap_spin(e , dlm, 2, geometry=self.geom_info, verbose=int(self.verbose))
            b = hp.map2alm_spin([Qdelen,Udelen],2)[1]
            hp.write_alm(fname,b)
            return b
    
    def delens_cl(self,i,th=True,iter=False):
        fname = os.path.join(self.basedir,f"delens_cl_{'g'if th else 'r'}{'' if self.lensed else 'gaus' }{'_i' if iter else ''}{'_s' if self.special_case else ''}_{i:04d}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            e = self.wf.mysims.sky.get_E(i)
            b = self.wf.mysims.sky.get_B(i)
            btemp = self.Btemp(i,th,iter)
            cld = hp.alm2cl(e,b-btemp)
            cll = hp.alm2cl(e,b)
            with open(fname,'wb') as f:
                pl.dump((cll,cld),f)
            return cll,cld
        
    def get_data_sp(self,iter:bool = False):
        fname = os.path.join(self.basedir,f"delens_arr_s{'_i' if iter else ''}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            lens_c_array = []
            delens_c_array = []
            for i in tqdm(range(100)):
                l,d = self.delens_cl(i, True,iter)
                lens_b, lens_c = bin_cmb_spectrum(l,5)
                delens_b, delens_c = bin_cmb_spectrum(d,5)
                lens_c_array.append(lens_c)
                delens_c_array.append(delens_c)
            lens_c_array = np.array(lens_c_array)
            delens_c_array = np.array(delens_c_array)
            data = {}
            data['b'] = lens_b
            data['lens'] = lens_c_array
            data['delens'] = delens_c_array
            pl.dump(data,open(fname,'wb'))
            return data

        

    def get_data(self,debias:bool = False,iter:bool = False):
        fname = os.path.join(self.basedir,f"delens_arr{int(debias)}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            lens_c_array = []
            delens_c_array_biased = []
            delens_c_array_theory = []
            for i in tqdm(range(100)):
                if debias:
                    l,d_nb = self.delens_cl(i, True)
                l,d_b = self.delens_cl(i, False)
                lens_b, lens_c = bin_cmb_spectrum(l,5)
                delens_biased_b, delens_biased_c = bin_cmb_spectrum(d_b,5)
                if debias:
                    delens_theory_b, delens_theory_c = bin_cmb_spectrum(d_nb,5)
                lens_c_array.append(lens_c)
                delens_c_array_biased.append(delens_biased_c)
                if debias:
                    delens_c_array_theory.append(delens_theory_c)
            delens_c_array_biased = np.array(delens_c_array_biased)
            if debias:
                delens_c_array_theory = np.array(delens_c_array_theory)
            lens_c_array = np.array(lens_c_array)
            if debias:
                bias = delens_c_array_theory.mean(axis=0) - delens_c_array_biased.mean(axis=0)
            data = {}
            data['b'] = lens_b
            data['lens'] = lens_c_array
            data['delens'] = delens_c_array_biased
            if debias:
                data['delens_theory'] = delens_c_array_theory
                data['bias'] = bias
            pl.dump(data,open(fname,'wb'))
            return data

    
