from numpy import arange
import sys
sys.path.append('../')

from dance.delens import Delens
from dance import mpi

basedir = '/mnt/sdceph/users/alonappan/DANCE_debug'

start_idx = 0
delens = Delens(basedir,2048,1,beta=0.35,lmin_ivf=2,lmax_ivf=4096,lmax_qlm=4096,verbose=0,special_case=True)
end_idx = 100

jobs = arange(start_idx,end_idx)

mpi.barrier()
for i in jobs[mpi.rank::mpi.size]:
    print(f"Rank {mpi.rank} is working on job {i}")
    eb = delens.delens_cl(i,th=True,iter=True)
    del eb
mpi.barrier()
