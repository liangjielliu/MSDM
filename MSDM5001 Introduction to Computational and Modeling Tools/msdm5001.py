from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 每个处理器生成一个随机整数
value = random.randint(1, 100)
print(f"Processor {rank} starts with value {value}")

# 排序逻辑：总共需要 size 次比较（冒泡排序方式）
for i in range(size):
    # 偶数轮：rank 0 与 1，2 与 3 等交换
    if i % 2 == 0:
        if rank % 2 == 0 and rank + 1 < size:
            # 与右边进程交换
            neighbor_value = comm.sendrecv(value, dest=rank + 1, source=rank + 1)
            if value > neighbor_value:
                value = neighbor_value
        elif rank % 2 == 1:
            # 接收左边进程的数据
            neighbor_value = comm.sendrecv(value, dest=rank - 1, source=rank - 1)
            if value < neighbor_value:
                value = neighbor_value
    # 奇数轮：rank 1 与 2，3 与 4 等交换
    else:
        if rank % 2 == 1 and rank + 1 < size:
            # 与右边进程交换
            neighbor_value = comm.sendrecv(value, dest=rank + 1, source=rank + 1)
            if value > neighbor_value:
                value = neighbor_value
        elif rank % 2 == 0 and rank - 1 >= 0:
            # 接收左边进程的数据
            neighbor_value = comm.sendrecv(value, dest=rank - 1, source=rank - 1)
            if value < neighbor_value:
                value = neighbor_value

# 确保所有进程排序完成后再开始输出
comm.Barrier()

# 按 rank 顺序逐个输出
for i in range(size):
    if rank == i:  # 当前进程的 rank 等于输出顺序
        print(f"Processor {rank} ends with value {value}")
    comm.Barrier()  # 确保其他进程等待当前进程输出完毕