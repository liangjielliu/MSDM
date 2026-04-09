from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 计算每个处理器的部分和
N = 100000  # 级数项数
local_sum = 0
for i in range(rank + 1, N + 1, size):
    local_sum += 1 / (i ** 2)

# 汇总结果
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Approximated value of π²/6: {total_sum}")