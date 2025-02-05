from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD  # Get the default communicator
rank = comm.Get_rank()  # Get the rank of the process
size = comm.Get_size()  # Get the total number of processes

print(f"Hello from rank {rank} out of {size} processes")

# Simple communication test
if rank == 0:
    data = "Hello, MPI!"
    comm.send(data, dest=1, tag=10)  # Sending data to rank 1
    print(f"Process {rank} sent data to Process 1")
elif rank == 1:
    received_data = comm.recv(source=0, tag=10)  # Receiving data from rank 0
    print(f"Process {rank} received data: {received_data}")

# Finalizing MPI
MPI.Finalize()