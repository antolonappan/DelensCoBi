from numpy import arange
import sys
sys.path.append('../')

from dance.qe import Reconstruct
from dance import mpi

basedir = '/mnt/sdceph/users/alonappan/DANCE_debug'

start_idx = 0
end_idx = 100

jobs = arange(start_idx,end_idx)

recon = Reconstruct(basedir,2048,1,model='aniso',Acb=1e-6,lmin_ivf=2,lmax_ivf=1024,lmax_qlm=1024,qe_key='a_p',verbose=1)

mpi.barrier()
for i in jobs[mpi.rank::mpi.size]:
    print(f"Rank {mpi.rank} is working on job {i}")
    eb = recon.get_qcl(i)
    del eb
mpi.barrier()
