import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.interpolate import InterpolatedUnivariateSpline
from getdist import MCSamples, plots
from dance.delens import Delens
from tqdm import tqdm
import healpy as hp
import os
from dance import mpi
from dance import utils
import pickle as pl


class Likelihood:

    def __init__(self,libdir,spectra,lmax=1000):
        cmb = spectra.cmb
        lb = spectra.blens
        lens = spectra.lens_t_b
        ld = spectra.bdlens
        delens = spectra.dlens_t_b 
        theory_unlens = cmb.get_unlensed_spectra(dl=False)
        theory_lens = cmb.get_lensed_spectra(dl=False)
        ell = np.arange(len(theory_unlens['ee']))
        
        fname = spectra.libdir.split('/')[-1][7:]
        self.basedir = os.path.join(libdir, f"lh{fname}")

        if mpi.rank == 0:
            os.makedirs(self.basedir,exist_ok=True)

        self.ee_unl_interp = InterpolatedUnivariateSpline(ell[2:],theory_unlens['ee'][2:],k=5)
        self.bb_unl_interp = InterpolatedUnivariateSpline(ell[2:],theory_unlens['bb'][2:],k=5)
        self.ee_lens_interp = InterpolatedUnivariateSpline(ell[2:],theory_lens['ee'][2:],k=5)
        self.bb_lens_interp = InterpolatedUnivariateSpline(ell[2:],theory_lens['bb'][2:],k=5)
        self.lb = lb
        self.ld = ld
        self.bl_sel = np.where(lb<lmax)[0]
        self.bd_sel = np.where(ld<lmax)[0]
        self.lens_t_mean = lens.mean(axis=0)[self.bl_sel]
        self.lens_t_std = lens.std(axis=0)[self.bl_sel]
        self.dlens_t_mean = delens.mean(axis=0)[self.bd_sel]
        self.dlens_t_std = delens.std(axis=0)[self.bd_sel]
        self.debias = spectra.debias
        self.bw = spectra.bw
        self.lmax = lmax

    def chi_sq_lens(self,beta):
        ee = self.ee_lens_interp(self.lb[self.bl_sel])
        bb = self.bb_lens_interp(self.lb[self.bl_sel])
        b = np.deg2rad(beta)
        eb = 0.5*(ee-bb) * np.sin(4*b)
        return np.sum((self.lens_t_mean - eb)**2 / self.lens_t_std**2)

    def chi_sq_delens(self,beta):
        ee = self.ee_unl_interp(self.lb[self.bd_sel])
        bb = self.bb_unl_interp(self.ld[self.bd_sel])
        b = np.deg2rad(beta)
        eb = 0.5*(ee-bb) * np.sin(4*b)
        return np.sum((self.dlens_t_mean - eb)**2 / self.dlens_t_std**2)

    def log_likelihood_lens(self,theta):
        return -0.5 * self.chi_sq_lens(theta)

    def log_likelihood_delens(self,theta):
        return -0.5 * self.chi_sq_delens(theta)

    def log_prior(self,theta):
        if 0 < theta < 0.5:
            return 0.0
        return -np.inf

    def log_probability_lens(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_lens(theta)

    def log_probability_delens(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_delens(theta)

    def get_pos_nwalkers_ndim(self):
        pos = [0.35] + 1e-1 * np.random.randn(32, 1)
        nwalkers, ndim = pos.shape
        return pos, nwalkers, ndim

    def get_lensed_samp(self):
        fname = os.path.join(self.basedir,f"lensed_d{int(self.debias)}_b{self.bw}_l{self.lmax}.pkl")
        if os.path.isfile(fname):
            samples = pl.load(open(fname,'rb'))
        else:
            pos, nwalkers, ndim = self.get_pos_nwalkers_ndim()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_lens)
            sampler.run_mcmc(pos, 5000)
            samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pl.dump(samples,open(fname,'wb'))
        names = ['beta']
        labels =  ['\\beta']
        return MCSamples(samples=samples, names = names, labels = labels, label='Lensed')

    def get_delensed_samp(self):
        fname = os.path.join(self.basedir,f"delensed_d{int(self.debias)}_b{self.bw}_l{self.lmax}.pkl")
        if os.path.isfile(fname):
            samples = pl.load(open(fname,'rb'))
        else:
            pos, nwalkers, ndim = self.get_pos_nwalkers_ndim()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_delens)
            sampler.run_mcmc(pos, 5000)
            samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pl.dump(samples,open(fname,'wb'))
        names = ['beta']
        labels =  ['\\beta']
        return MCSamples(samples=samples, names = names, labels = labels, label='Delensed')

    def plot_compare(self):
        lsamples = self.get_lensed_samp()
        dsamples = self.get_delensed_samp()
        g = plots.get_subplot_plotter(width_inch=4)
        g.triangle_plot([lsamples,dsamples], filled=True)
        plt.show()

class Spectra:

    def __init__(self,libdir,nside,nlevp,model,beta,lmin_ivf,lmax_ivf,lmax_qlm,qe_key,lmin_delens,lmax_delens,debias,bw):
        self.delens_r = Delens(libdir,nside,nlevp,True,model,beta=beta,lmin_ivf=lmin_ivf,lmax_ivf=lmax_ivf,lmax_qlm=lmax_qlm,qe_key=qe_key,lmin_delens=lmin_delens,lmax_delens=lmax_delens)
        self.delens_g = Delens(libdir,nside,nlevp,False,model,beta=beta,lmin_ivf=lmin_ivf,lmax_ivf=lmax_ivf,lmax_qlm=lmax_qlm,qe_key=qe_key,lmin_delens=lmin_delens,lmax_delens=lmax_delens)
        
        self.cmb = self.delens_r.wf.cmb
        self.libdir = os.path.join(libdir,f'spectra_N{nside}_n{nlevp}_m{model}_b{beta}_{lmin_ivf}{lmax_ivf}{lmax_qlm}{lmin_delens}{lmax_delens}')
        if mpi.rank == 0:
            os.makedirs(self.libdir,exist_ok=True)

        fname_bias = os.path.join(self.libdir,'gauss_bias.pkl')
        if os.path.isfile(fname_bias):
            cl_diff = pl.load(open(fname_bias,'rb'))
        else:
            sky_g = self.delens_g.wf.mysims.sky
            cl_diff = []
            for i in tqdm(range(50)):
                cl_g = self.delens_g.delens_cl(i)
                cl_d_g = hp.alm2cl(sky_g.get_E(i),sky_g.get_B(i))
                cl_diff.append(cl_g - cl_d_g)
            cl_diff = np.array(cl_diff).mean(axis=0)
            pl.dump(cl_diff,open(fname_bias,'wb'))


        fname_lens = os.path.join(self.libdir,'lens.pkl')
        if os.path.isfile(fname_lens):
            lens_t = pl.load(open(fname_lens,'rb'))
        else:
            sky_l = self.delens_r.wf.mysims.sky
            lens_t = []
            for i in tqdm(range(100)):
                cl_d_l = hp.alm2cl(sky_l.get_E(i),sky_l.get_B(i))
                lens_t.append(cl_d_l)
            lens_t = np.array(lens_t)
            pl.dump(lens_t,open(fname_lens,'wb'))
        

        fname_delens = os.path.join(self.libdir,'delens.pkl')
        if os.path.isfile(fname_delens):
            dlens_t = pl.load(open(fname_delens,'rb'))
        else:
            dlens_t = []
            for i in tqdm(range(100)):
                cl = self.delens_r.delens_cl(i,recon=True)
                dlens_t.append(cl)
            dlens_t = np.array(dlens_t)
            pl.dump(dlens_t,open(fname_delens,'wb'))
        

        fl = hp.gauss_beam(np.radians(2/60),len(lens_t[0])-1)

        lens_t_b = []
        for i in tqdm(range(100)):
            self.blens, cb_lens = utils.bin_cmb_spectrum(lens_t[i]/fl**2,bw)
            lens_t_b.append(cb_lens)
        self.lens_t_b = np.array(lens_t_b)


        dlens_t_b = []
        for i in tqdm(range(100)):
            cllll = dlens_t[i]
            if debias:
                cllll = dlens_t[i] - cl_diff
            self.bdlens, cb_dlens = utils.bin_cmb_spectrum(cllll/fl**2,bw)
            dlens_t_b.append(cb_dlens)
        self.dlens_t_b = np.array(dlens_t_b)
        
        self.eb_lens = self.delens_r.recon.cmb.get_cb_lensed_spectra(0.35,dl=False)['eb']
        self.eb_unlens = self.delens_r.recon.cmb.get_cb_unlensed_spectra(0.35,dl=False)['eb']

        self.dll = self.blens*(self.blens+1)/2/np.pi
        self.dbl = self.bdlens*(self.bdlens+1)/2/np.pi
        self.l = np.arange(len(self.eb_lens))
        self.dl = self.l*(self.l+1)/2/np.pi
        self.debias = debias
        self.bw = bw

    def plot_spectra(self):
        plt.figure(figsize=(6,6))
        plt.loglog(self.eb_lens*self.dl,label='Signal',c='k',lw=2,ls='--')
        plt.errorbar(self.blens,self.lens_t_b.mean(axis=0)*self.dll,self.lens_t_b.std(axis=0)*self.dll,fmt='o',ms=10,elinewidth=6,label='Lensed')
        plt.errorbar(self.bdlens,self.dlens_t_b.mean(axis=0)*self.dbl,self.dlens_t_b.std(axis=0)*self.dbl,fmt='o', ms=5,elinewidth=2,label='Delensed')
        plt.ylim(1e-7,1e-1)
        plt.xlim(2,200)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.ylabel(r'$\ell(\ell+1)C^{EB}_{\ell}/2\pi$',fontsize=20)
        plt.legend(fontsize=20)

    def plot_peaks(self):
        ratio = (self.eb_lens*self.dl) / (self.eb_unlens*self.dl)
        data_ratio = (self.lens_t_b.mean(axis=0)*self.dll) / (self.dlens_t_b.mean(axis=0)*self.dbl)
        std_ratio = np.abs(data_ratio * np.sqrt((self.lens_t_b.std(axis=0)/self.lens_t_b.mean(axis=0))**2 + (self.dlens_t_b.std(axis=0)/self.dlens_t_b.mean(axis=0))**2))
        plt.figure(figsize=(6,6))
        plt.plot(ratio,label=r'$C^{EB}_{\ell,lensed}/C^{EB}_{\ell,unlensed}$',c='k',lw=2,ls='--')
        plt.plot(self.bdlens,data_ratio,label=r'$C^{EB}_{\ell,lensed}/C^{EB}_{\ell,delensed}$',c='r',lw=2)
        plt.fill_between(self.bdlens,data_ratio-std_ratio,data_ratio+std_ratio,alpha=0.4,color='r')
        plt.ylim(0.5,1.5)
        plt.xlim(400,2000)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.legend(fontsize=20)



        