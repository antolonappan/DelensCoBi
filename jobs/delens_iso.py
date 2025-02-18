from numpy import arange
import sys
sys.path.append('../')

from dance.delens import Delens
from dance import mpi

basedir = '/mnt/sdceph/users/alonappan/DANCE'

lensed = True

start_idx = 0
if lensed:
    delens = Delens(basedir,2048,1,True,"iso",beta=0.35,lmin_ivf=2,lmax_ivf=4096,lmax_qlm=4096,qe_key="p_p",lmin_delens=50,lmax_delens=4096) 
    end_idx = 100
else:
    delens = Delens(basedir,2048,1,False,"iso",beta=0.35,lmin_ivf=2,lmax_ivf=4096,lmax_qlm=4096,qe_key="p_p",lmin_delens=50,lmax_delens=4096) 
    end_idx = 50

jobs = arange(start_idx,end_idx)

mpi.barrier()
for i in jobs[mpi.rank::mpi.size]:
    print(f"Rank {mpi.rank} is working on job {i}")
    eb = delens.delens_cl(i,th=True)
    del eb
mpi.barrier()
