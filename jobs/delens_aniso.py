from numpy import arange
import sys
sys.path.append('../')

from dance.delens import Delens
from dance.qe import Reconstruct
from dance import mpi

basedir = '/mnt/sdceph/users/alonappan/DANCE'

delensed = True


start_idx = 90
end_idx = 100

jobs = arange(start_idx,end_idx)

if delensed:
    delens = Delens(basedir,2048,1,True,"aniso",Acb=1e-6,lmin_ivf=2,lmax_ivf=4096,lmax_qlm=4096,qe_key="p_p",lmin_delens=50,lmax_delens=4096)
    recon = Reconstruct(basedir,2048,1,True,"aniso",Acb=1e-6,lmin_ivf=2,lmax_ivf=4096,lmax_qlm=4096,qe_key="a_p",delens=delens)
else:
    recon = Reconstruct(basedir,2048,1,True,"aniso",Acb=1e-6,lmin_ivf=2,lmax_ivf=4096,lmax_qlm=4096,qe_key="a_p")

mpi.barrier()
for i in jobs[mpi.rank::mpi.size]:
    print(f"Rank {mpi.rank} is working on job {i}")
    cl = recon.get_qcl(i)
mpi.barrier()
