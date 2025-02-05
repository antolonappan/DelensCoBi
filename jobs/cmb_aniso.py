from numpy import arange
import sys
sys.path.append('../')

from dance.simulations import CMB
from dance import mpi

basedir = '/mnt/home/alonappan/scratch/DANCE'
cmb = CMB(basedir,2048,'aniso',Acb=1e-6)

start_idx = 0
end_idx = 300

jobs = arange(start_idx,end_idx)

mpi.barrier()
for i in jobs[mpi.rank::mpi.size]:
    print(f"Rank {mpi.rank} is working on job {i}")
    qu = cmb.get_QU(i)
    del qu
mpi.barrier()