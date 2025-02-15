from numpy import arange
import sys
sys.path.append('../')

from dance.qe import Reconstruct
from dance import mpi

basedir = '/mnt/sdceph/users/alonappan/DANCE'
recon = Reconstruct(basedir,1024,"iso",beta=0.35,lmin_ivf=2,lmax_ivf=3000,lmax_qlm=3000,qe_key="p_p")


start_idx = 2
end_idx = 300

jobs = arange(start_idx,end_idx)

mpi.barrier()
for i in jobs[mpi.rank::mpi.size]:
    print(f"Rank {mpi.rank} is working on job {i}")
    qlm = recon.get_qlm(i)
    n0 = recon.get_n0(i)
    del (qlm,n0)
mpi.barrier()