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
from dance.data import CAMB_INI
import camb
import hashlib

def compute_delens_spectra(basedir,spec):
    dname = os.path.join(basedir, 'Data')
    os.makedirs(dname,exist_ok=True)
    
    N0 = []
    for i in tqdm(range(100)):
        N0.append(spec.delens_r.recon.get_n0(i))
    N0 = np.array(N0).mean(axis=0)

    fname = os.path.join(dname, f'cmbspectra_N0_{hashlib.md5(N0).hexdigest()}.pkl')

    if os.path.isfile(fname):
        cl_dict = pl.load(open(fname,'rb'))
        return cl_dict
    else:
        lmax = len(N0) - 1
        CAMB_INI.directory = basedir
        params   = CAMB_INI.data
        params.set_for_lmax(lmax-150)
        results  = camb.get_results(params)


        cl_pp = spec.delens_r.recon.cmb.cl_pp()[:len(N0)]
        w = lambda ell: ell ** 2 * (ell + 1) ** 2 * 0.5 / np.pi
        ellp = np.arange(len(N0))
        cl_pp_res = cl_pp*(1 - (cl_pp/(cl_pp+N0+1e-30)))
        cl_pp_res[:2] = 0
        delensed = results.get_lensed_cls_with_spectrum(cl_pp_res*w(ellp),lmax=4096,CMB_unit='muK',raw_cl=True)
        cl_dict = {}
        cl_dict['ee'] = delensed[:,1]
        cl_dict['bb'] = delensed[:,2]
        pl.dump(cl_dict,open(fname,'wb'))
        return cl_dict



class Likelihood:

    def __init__(self,libdir,spectra,lmax=1000,use_delens=False,fix_alens=True,use_bb=False):
        cmb = spectra.cmb
        b_eb_lens = spectra.blens
        lens_eb = spectra.lens_t_b
        b_eb_dlens = spectra.bdlens
        delens_eb = spectra.dlens_t_b
        b_bb_lens = spectra.blensb
        lens_bb = spectra.lensb_t_b
        b_bb_dlens = spectra.bdlensb
        delens_bb = spectra.dlensb_t_b


        theory_unlens = cmb.get_unlensed_spectra(dl=False)
        theory_lens = cmb.get_lensed_spectra(dl=False)
        theory_delens = compute_delens_spectra(libdir,spectra)
        ell = np.arange(len(theory_unlens['ee']))
        elld = np.arange(len(theory_delens['ee']))
        
        fname = spectra.libdir.split('/')[-1][7:]
        self.basedir = os.path.join(libdir, f"lh{fname}")

        if mpi.rank == 0:
            os.makedirs(self.basedir,exist_ok=True)

        self.ee_unl_interp = InterpolatedUnivariateSpline(ell[2:],theory_unlens['ee'][2:],k=5)
        self.bb_unl_interp = InterpolatedUnivariateSpline(ell[2:],theory_unlens['bb'][2:],k=5)
        self.ee_lens_interp = InterpolatedUnivariateSpline(ell[2:],theory_lens['ee'][2:],k=5)
        self.bb_lens_interp = InterpolatedUnivariateSpline(ell[2:],theory_lens['bb'][2:],k=5)
        self.ee_delens_interp = InterpolatedUnivariateSpline(elld[2:],theory_delens['ee'][2:],k=5)
        self.bb_delens_interp = InterpolatedUnivariateSpline(elld[2:],theory_delens['bb'][2:],k=5)

        l_eb_sel = np.where(b_eb_lens < lmax)[0]
        d_eb_sel = np.where(b_eb_dlens < lmax)[0]
        l_bb_sel = np.where(b_bb_lens < lmax)[0]
        d_bb_sel = np.where(b_bb_dlens < lmax)[0]

        self.l_eb = b_eb_lens[l_eb_sel]
        self.d_eb = b_eb_dlens[d_eb_sel]
        self.l_bb = b_bb_lens[l_bb_sel]
        self.d_bb = b_bb_dlens[d_bb_sel]
         
        self.lens_eb_mean = lens_eb.mean(axis=0)[l_eb_sel]
        self.lens_eb_std = lens_eb.std(axis=0)[l_eb_sel]
        self.delens_eb_mean = delens_eb.mean(axis=0)[d_eb_sel]
        self.delens_eb_std = delens_eb.std(axis=0)[d_eb_sel]
        self.lens_bb_mean = lens_bb.mean(axis=0)[l_bb_sel]
        self.lens_bb_std = lens_bb.std(axis=0)[l_bb_sel]
        self.delens_bb_mean = delens_bb.mean(axis=0)[d_bb_sel]
        self.delens_bb_std = delens_bb.std(axis=0)[d_bb_sel]


        self.debias = spectra.debias
        self.bw = spectra.bw
        self.lmax = lmax
        self.use_delens = use_delens
        self.fix_alens = fix_alens
        self.use_bb = use_bb
    
    def theory_eb(self,beta,Alens,ell=None,which='lens'):
        if ell is None:
            ell = np.arange(0,self.lmax)
        if which == 'lens':
            ee = self.ee_lens_interp(ell)
            bb = self.bb_lens_interp(ell)
        elif which == 'delens':
            ee = self.ee_delens_interp(ell)
            bb = self.bb_delens_interp(ell)
        elif which == 'unlens':
            ee = self.ee_unl_interp(ell)
            bb = self.bb_unl_interp(ell)
        else:
            raise ValueError('which must be lens, delens or unlens')
        b = np.deg2rad(beta)
        eb = 0.5*(ee-(Alens*bb)) * np.sin(4*b)
        return eb
    
    def theory_bb(self,beta,Alens,ell=None,which='lens'):
        if ell is None:
            ell = np.arange(0,self.lmax)
        if which == 'lens':
            ee = self.ee_lens_interp(ell)
            bb = self.bb_lens_interp(ell)
        elif which == 'delens':
            ee = self.ee_delens_interp(ell)
            bb = self.bb_delens_interp(ell)
        elif which == 'unlens':
            ee = self.ee_unl_interp(ell)
            bb = self.bb_unl_interp(ell)
        else:
            raise ValueError('which must be lens, delens or unlens')
        b = np.deg2rad(2*beta)
        return (ee * np.sin(b)**2) + ((Alens*bb) * np.cos(b)**2)
    
    def chi_sq_eb(self,beta,Alens,which='lens'):
        if which == 'lens':
            eb = self.theory_eb(beta,Alens,self.l_eb,which='lens')
            return np.sum((self.lens_eb_mean - eb)**2 / self.lens_eb_std**2)
        else:
            if self.use_delens:
                eb = self.theory_eb(beta,Alens,self.d_eb,which='delens')
            else:
                eb = self.theory_eb(beta,Alens,self.d_eb,which='unlens')
            return np.sum((self.delens_eb_mean - eb)**2 / self.delens_eb_std**2)
        
    def chi_sq_bb(self,beta,Alens,which='lens'):
        if which == 'lens':
            bb = self.theory_bb(beta,Alens,self.l_bb,which='lens')
            return np.sum((self.lens_bb_mean - bb)**2 / self.lens_bb_std**2)
        else:
            if self.use_delens:
                bb = self.theory_bb(beta,Alens,self.d_bb,which='delens')
            else:
                bb = self.theory_bb(beta,Alens,self.d_bb,which='unlens')
            return np.sum((self.delens_bb_mean - bb)**2 / self.delens_bb_std**2)

    def log_likelihood_lens(self,theta):
        if self.fix_alens:
            beta = theta
            Alens = 1
        else:
            beta, Alens = theta
        if self.use_bb:
            chisq = self.chi_sq_eb(beta,Alens,which='lens') + self.chi_sq_bb(beta,Alens,which='lens')
        else:
            chisq = self.chi_sq_eb(beta,Alens,which='lens')
        return -0.5 * chisq
    
    def log_likelihood_delens(self,theta):
        if self.fix_alens:
            beta = theta
            Alens = 1
            if self.use_bb:
                chisq = self.chi_sq_eb(beta,Alens,which='delens') + self.chi_sq_bb(beta,Alens,which='delens')
            else:
                chisq = self.chi_sq_eb(beta,Alens,which='delens')
            return -0.5 * chisq
        else:
            return self.log_likelihood_lens(theta)


    def log_prior(self,theta):
        if self.fix_alens:
            beta = theta
            if 0 < beta < 0.5:
                return 0.0
        else:
            beta, Alens = theta
            if 0 < beta < 0.5 and 0 < Alens < 2:
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
        if self.fix_alens:
            pos = [0.35] + 1e-1 * np.random.randn(32, 1)
        else:
            pos = np.concatenate([0.35 + 1e-1 * np.random.randn(32, 1), 1 + .5 * np.random.randn(32, 1)], axis=1)
        nwalkers, ndim = pos.shape
        return pos, nwalkers, ndim

    def get_lensed_samp(self,shift=0.0):
        # fname = os.path.join(self.basedir,f"lensed_d{int(self.debias)}_b{self.bw}_l{self.lmax}.pkl")
        # if os.path.isfile(fname):
        #     samples = pl.load(open(fname,'rb'))
        # else:
        pos, nwalkers, ndim = self.get_pos_nwalkers_ndim()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_lens)
        sampler.run_mcmc(pos, 5000, progress=True)
        samples = sampler.get_chain(discard=100, thin=15, flat=True) + shift
        return samples
        #    pl.dump(samples,open(fname,'wb'))
        if self.fix_alens:
            names = ['beta']
            labels =  ['\\beta']
        else:
            names = ['beta','Alens']
            labels =  ['\\beta','A_{lens}']
        return MCSamples(samples=samples, names = names, labels = labels, label='Lensed')

    def get_delensed_samp(self,shift=0.0):
        # fname = os.path.join(self.basedir,f"delensed_d{int(self.debias)}_b{self.bw}_l{self.lmax}{'' if not self.use_delens else '_thd'}.pkl")
        # if os.path.isfile(fname):
        #     samples = pl.load(open(fname,'rb')) + shift
        # else:
        pos, nwalkers, ndim = self.get_pos_nwalkers_ndim()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_delens)
        sampler.run_mcmc(pos, 5000, progress=True)
        samples = sampler.get_chain(discard=100, thin=15, flat=True) + shift
        return samples
            # pl.dump(samples,open(fname,'wb'))
        if self.fix_alens:
            names = ['beta']
            labels =  ['\\beta']
        else:
            names = ['beta','Alens']
            labels =  ['\\beta','A_{lens}']
        return MCSamples(samples=samples, names = names, labels = labels, label=f"{'Delensed(debiased)'if self.debias else 'Delensed(biased)'}")

    def plot_compare(self,shift=0.0,savename=None):
        lsamples = self.get_lensed_samp()
        dsamples = self.get_delensed_samp(shift=shift)
        print('lensed',lsamples.getInlineLatex('beta',limit=1))
        print('delensed',dsamples.getInlineLatex('beta',limit=1))
        g = plots.get_subplot_plotter(width_inch=4)
        g.triangle_plot([lsamples,dsamples], filled=True)
        plt.axvline(0.35, c='k', ls='--')
        if savename is not None:
            plt.savefig(savename,bbox_inches='tight',dpi=300)

class Spectra:

    def __init__(self,libdir,nside,nlevp,model,beta,lmin_ivf,lmax_ivf,lmax_qlm,qe_key,lmin_delens,lmax_delens,debias,bw,th=False):
        self.delens_r = Delens(libdir,nside,nlevp,True,model,beta=beta,lmin_ivf=lmin_ivf,lmax_ivf=lmax_ivf,lmax_qlm=lmax_qlm,qe_key=qe_key,lmin_delens=lmin_delens,lmax_delens=lmax_delens)
        self.delens_g = Delens(libdir,nside,nlevp,False,model,beta=beta,lmin_ivf=lmin_ivf,lmax_ivf=lmax_ivf,lmax_qlm=lmax_qlm,qe_key=qe_key,lmin_delens=lmin_delens,lmax_delens=lmax_delens)
        
        self.cmb = self.delens_r.wf.cmb
        self.libdir = os.path.join(libdir,f'spectra_N{nside}_n{nlevp}_m{model}_b{beta}_{lmin_ivf}{lmax_ivf}{lmax_qlm}{lmin_delens}{lmax_delens}')
        if mpi.rank == 0:
            os.makedirs(self.libdir,exist_ok=True)
        
        # Gaussian bias EB
        fname_bias = os.path.join(self.libdir,'gauss_bias.pkl' if not th else 'th_bias.pkl')
        if os.path.isfile(fname_bias):
            cl_diff = pl.load(open(fname_bias,'rb'))
        else:
            sky_g = self.delens_g.wf.mysims.sky
            cl_diff = []
            for i in tqdm(range(50)):
                cl_g = self.delens_g.delens_cl(i,th=th)
                cl_d_g = hp.alm2cl(sky_g.get_E(i),sky_g.get_B(i))
                cl_diff.append(cl_g - cl_d_g)
            cl_diff = np.array(cl_diff).mean(axis=0)
            pl.dump(cl_diff,open(fname_bias,'wb'))
        
        # Gaussian bias BB
        fname_bias_b = os.path.join(self.libdir,'gauss_bias_b.pkl' if not th else 'th_bias_b.pkl')
        if os.path.isfile(fname_bias_b):
            cl_diff_b = pl.load(open(fname_bias_b,'rb'))
        else:
            sky_g = self.delens_g.wf.mysims.sky
            cl_diff_b = []
            for i in tqdm(range(50)):
                cl_g = hp.alm2cl(self.delens_g.delens(i)[1])
                cl_d_g = hp.alm2cl(sky_g.get_B(i))
                cl_diff_b.append(cl_g - cl_d_g)
            cl_diff_b = np.array(cl_diff_b).mean(axis=0)
            pl.dump(cl_diff_b,open(fname_bias_b,'wb'))
        

        # Lens EB
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


        # Lens BB
        fname_lens_b = os.path.join(self.libdir,'lens_b.pkl')
        if os.path.isfile(fname_lens_b):
            lensb_t = pl.load(open(fname_lens_b,'rb'))
        else:
            lensb_t = []
            sky_l = self.delens_r.wf.mysims.sky
            for i in tqdm(range(100)):
                cl = hp.alm2cl(sky_l.get_B(i))
                lensb_t.append(cl)
            lensb_t = np.array(lens_t_b)
            pl.dump(lensb_t,open(fname_lens_b,'wb'))
                
        # Delens EB
        fname_delens = os.path.join(self.libdir,'delens.pkl' if not th else 'th_delens.pkl')
        if os.path.isfile(fname_delens):
            dlens_t = pl.load(open(fname_delens,'rb'))
        else:
            dlens_t = []
            for i in tqdm(range(100)):
                cl = self.delens_r.delens_cl(i,th=th)
                dlens_t.append(cl)
            dlens_t = np.array(dlens_t)
            pl.dump(dlens_t,open(fname_delens,'wb'))


        # Delens BB
        fname_delens_b = os.path.join(self.libdir,'delens_b.pkl' if not th else 'th_delens_b.pkl')
        if os.path.isfile(fname_delens_b):
            dlensb_t = pl.load(open(fname_delens_b,'rb'))
        else:
            dlensb_t = []
            for i in tqdm(range(100)):
                cl = hp.alm2cl(self.delens_r.delens(i)[1])
                dlensb_t.append(cl)
            dlensb_t = np.array(dlensb_t)
            pl.dump(dlensb_t,open(fname_delens_b,'wb'))

        
        fl = hp.gauss_beam(np.radians(2/60),len(lens_t[0])-1)
        self.fl = fl

        # Lens EB binned
        lens_t_b = []
        for i in tqdm(range(100)):
            self.blens, cb_lens = utils.bin_cmb_spectrum(lens_t[i]/fl**2,bw)
            lens_t_b.append(cb_lens)
        self.lens_t_b = np.array(lens_t_b)

        # Delens EB binned
        dlens_t_b = []
        for i in tqdm(range(100)):
            cllll = dlens_t[i]
            if debias:
                cllll = dlens_t[i] - cl_diff
            self.bdlens, cb_dlens = utils.bin_cmb_spectrum(cllll/fl**2,bw)
            dlens_t_b.append(cb_dlens)
        self.dlens_t_b = np.array(dlens_t_b)
        
        # Lens BB binned
        lensb_t_b = []
        for i in tqdm(range(100)):
            cllll = lensb_t[i]
            self.blensb, cb_lensb = utils.bin_cmb_spectrum(cllll/fl**2,bw)
            lensb_t_b.append(cb_lensb)
        self.lensb_t_b = np.array(lensb_t_b)

        # Delens BB binned
        dlensb_t_b = []
        for i in tqdm(range(100)):
            cllll = dlensb_t[i]
            # if debias:
            #     cllll = dlensb_t[i] - cl_diff_b
            self.bdlensb, cb_dlensb = utils.bin_cmb_spectrum(cllll/fl**2,bw)
            dlensb_t_b.append(cb_dlensb)
        self.dlensb_t_b = np.array(dlensb_t_b)

        
        self.eb_lens = self.delens_r.recon.cmb.get_cb_lensed_spectra(0.35,dl=False)['eb']
        self.eb_unlens = self.delens_r.recon.cmb.get_cb_unlensed_spectra(0.35,dl=False)['eb']
        self.bb_lens = self.delens_r.recon.cmb.get_cb_lensed_spectra(0.35,dl=False)['bb']

        self.dll = self.blens*(self.blens+1)/2/np.pi
        self.dbl = self.bdlens*(self.bdlens+1)/2/np.pi
        self.dbdl = self.bdlensb*(self.bdlensb+1)/2/np.pi
        self.l = np.arange(len(self.eb_lens))
        self.dl = self.l*(self.l+1)/2/np.pi
        self.debias = debias
        self.bw = bw

    def plot_spectra(self,savename=None):
        plt.figure(figsize=(6,6))
        plt.loglog(self.eb_lens*self.dl,label='Signal',c='k',lw=2,ls='--')
        plt.errorbar(self.blens,self.lens_t_b.mean(axis=0)*self.dll,self.lens_t_b.std(axis=0)*self.dll,fmt='o',ms=10,elinewidth=6,label='Lensed')
        plt.errorbar(self.bdlens,self.dlens_t_b.mean(axis=0)*self.dbl,self.dlens_t_b.std(axis=0)*self.dbl,fmt='o', ms=5,elinewidth=2,label='Delensed')
        plt.ylim(1e-7,1e-1)
        plt.xlim(2,200)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.ylabel(r'$\ell(\ell+1)C^{EB}_{\ell}/2\pi$',fontsize=20)
        plt.legend(fontsize=20)
        if savename is not None:
            plt.savefig(savename,bbox_inches='tight',dpi=300)

    def plot_peaks(self, return_arr=False):
        ratio = (self.eb_lens*self.dl) / (self.eb_unlens*self.dl)
        data_ratio = (self.lens_t_b.mean(axis=0)*self.dll) / (self.dlens_t_b.mean(axis=0)*self.dbl)
        std_ratio = np.abs(data_ratio * np.sqrt((self.lens_t_b.std(axis=0)/self.lens_t_b.mean(axis=0))**2 + (self.dlens_t_b.std(axis=0)/self.dlens_t_b.mean(axis=0))**2))
        if return_arr:
            d = {}
            d['ratio'] = ratio
            d['dratio'] =[self.bdlens,data_ratio]
            d['dsratio'] = [self.bdlens,data_ratio-std_ratio,data_ratio+std_ratio]
            return d
        plt.figure(figsize=(6,6))
        plt.plot(ratio,label=r'$C^{EB}_{\ell,lensed}/C^{EB}_{\ell,unlensed}$',c='k',lw=2,ls='--')
        plt.plot(self.bdlens,data_ratio,label=r'$C^{EB}_{\ell,lensed}/C^{EB}_{\ell,delensed}$',c='r',lw=2)
        plt.fill_between(self.bdlens,data_ratio-std_ratio,data_ratio+std_ratio,alpha=0.4,color='r')
        plt.ylim(0.5,1.5)
        plt.xlim(400,2000)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.legend(fontsize=20)



        