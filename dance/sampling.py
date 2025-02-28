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

    def __init__(self,delens,lmax=1000,debias=False):
        self.basedir = delens.basedir
        self.lmax = lmax
        cmb = delens.wf.cmb
        fwhm = delens.wf.mysims.sky.noise.fwhm(unit='rad')
        theory_lens = cmb.get_lensed_spectra(dl=False)
        ell = np.arange(len(theory_lens['ee']))
        bl = hp.gauss_beam(fwhm,lmax=len(ell)-1)
    
        if mpi.rank == 0:
            os.makedirs(self.basedir,exist_ok=True)
        self.ee_lens_interp = InterpolatedUnivariateSpline(ell[2:],theory_lens['ee'][2:],k=5)
        self.bb_lens_interp = InterpolatedUnivariateSpline(ell[2:],theory_lens['bb'][2:],k=5)
        fl = InterpolatedUnivariateSpline(ell[2:],bl[2:],k=5)

        data = delens.get_data(debias=debias)

        b = data['b']

        sel = np.where(b < lmax)[0]
        
        self.b = b[sel]
        self.lens_eb_mean = data['lens'].mean(axis=0)[sel]
        self.lens_eb_std = data['lens'].std(axis=0)[sel]
        self.delens_eb_mean = data['delens'].mean(axis=0)[sel]
        self.delens_eb_std = data['delens'].std(axis=0)[sel]
        self.fb = fl(self.b)
        self.debias = debias
        if debias:
            self.bias = data['bias'][sel]
        else:
            self.bias = None

    
    def theory_eb(self,beta,ell=None):
        if ell is None:
            ell = np.arange(0,self.lmax)
        ee = self.ee_lens_interp(ell)
        bb = self.bb_lens_interp(ell)
        b = np.deg2rad(beta)
        eb = 0.5*(ee-bb) * np.sin(4*b)
        return eb
    
    def chi_sq(self,beta,which,debias=False):
        eb = self.theory_eb(beta,self.b)
        if which == 'l':
            return np.sum((self.lens_eb_mean/self.fb**2 - eb)**2 / self.lens_eb_std**2)
        elif which == 'd':
            if debias:
                assert self.bias is not None, 'Bias is not available in data'
                data_eb = (self.delens_eb_mean + self.bias)/self.fb**2
            else:
                data_eb = self.delens_eb_mean/self.fb**2
            return np.sum((data_eb - eb)**2 / self.delens_eb_std**2)
        else:
            raise ValueError('which must be either l or d')
        
    def log_likelihood(self,theta,which,debias):
        chisq = self.chi_sq(theta,which,debias)
        return -0.5 * chisq

    def log_prior(self,theta):
        beta = theta
        if 0 < beta < 0.5:
            return 0.0
        return -np.inf

    def log_probability(self,theta,which,debias):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta,which,debias)
    
    def log_prob_lens(self,theta):
        return self.log_probability(theta,'l',False)
    
    def log_prob_delens(self,theta,debias):
        return self.log_probability(theta,'d',debias)

    def get_pos_nwalkers_ndim(self):
        pos = [0.35] + 1e-1 * np.random.randn(32, 1)
        nwalkers, ndim = pos.shape
        return pos, nwalkers, ndim

    def get_lensed_samp(self,getdist=True):
        fname = os.path.join(self.basedir,f"lensed_l{self.lmax}.pkl")
        if os.path.isfile(fname):
            samples = pl.load(open(fname,'rb'))
        else:
            pos, nwalkers, ndim = self.get_pos_nwalkers_ndim()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob_lens)
            sampler.run_mcmc(pos, 5000, progress=True)
            samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pl.dump(samples,open(fname,'wb'))
        if getdist:
            names = ['beta']
            labels =  ['\\beta']
            return MCSamples(samples=samples, names = names, labels = labels, label='Lensed')
        else:
            return samples
        
    def get_delensed_samp(self,debias=False,getdist=True):
        fname = os.path.join(self.basedir,f"delensed_l{self.lmax}_{int(debias)}.pkl")
        if os.path.isfile(fname):
            samples = pl.load(open(fname,'rb'))
        else:
            pos, nwalkers, ndim = self.get_pos_nwalkers_ndim()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob_delens, args=[debias])
            sampler.run_mcmc(pos, 5000, progress=True)
            samples = sampler.get_chain(discard=100, thin=15, flat=True) 
            pl.dump(samples,open(fname,'wb'))
        if getdist:
            names = ['beta']
            labels =  ['\\beta']
            if self.debias:
                if debias:
                    label = 'Delensed (Debiased)'
                else:
                    label = 'Delensed (Biased)'
            else:
                label = 'Delensed'
            return MCSamples(samples=samples, names = names, labels = labels, label=label)
        else:
            return samples

    def plot_compare(self):
        ls = self.get_lensed_samp()
        ds_b = self.get_delensed_samp()
        if self.debias:
            ds_db = self.get_delensed_samp(debias=True)
        g = plots.get_subplot_plotter(width_inch=4)
        if self.debias:
            g.triangle_plot([ls,ds_b,ds_db], filled=True)
        else:
            g.triangle_plot([ls,ds_b], filled=True)
        plt.axvline(0.35, c='k', ls='--')

    def get_limits(self):
        ls = self.get_lensed_samp()
        print('Lensed:', ls.getInlineLatex('beta',limit=1))
        ds_b = self.get_delensed_samp()
        print('Delensed (Biased):', ds_b.getInlineLatex('beta',limit=1))
        if self.debias:
            ds_db = self.get_delensed_samp(debias=True)
            print('Delensed (Debiased):', ds_db.getInlineLatex('beta',limit=1))
        

