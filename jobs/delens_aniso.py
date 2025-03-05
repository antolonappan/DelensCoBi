from numpy import arange
import sys
sys.path.append('../')

from dance.delens import Delens
from dance.qe import Reconstruct
from dance import mpi

basedir = '/mnt/sdceph/users/alonappan/DANCE_debug'


start_idx = 0
end_idx = 100

jobs = arange(start_idx,end_idx)


delens = Delens(basedir,2048,1,model='aniso',Acb=1e-6,lmin_ivf=2,lmax_ivf=4096,lmax_qlm=4096,qe_key='p_p',verbose=1,special_case=False)
recon = Reconstruct(basedir,2048,1,model='aniso',Acb=1e-6,lmin_ivf=2,lmax_ivf=1024,lmax_qlm=1024,qe_key='a_p',verbose=1,delens=delens)


mpi.barrier()
for i in jobs[mpi.rank::mpi.size]:
    print(f"Rank {mpi.rank} is working on job {i}")
    cl = recon.get_qcl(i)
mpi.barrier()
