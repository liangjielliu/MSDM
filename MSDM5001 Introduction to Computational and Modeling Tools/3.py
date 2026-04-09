from mpi4py import MPI
import numpy as np
import random

# 初始化 MPI 环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 原始图的邻接表
original_graph = {
    7: [12, 17],
    9: [15, 16],
    12: [15, 22, 7],
    15: [12, 17, 9, 22],
    16: [22, 9],
    17: [24, 15, 22, 7],
    22: [12, 16, 17, 15, 24],
    24: [17, 22]
}

# 将图节点编号映射到连续的 MPI rank
nodes = list(original_graph.keys())
node_mapping = {node: idx for idx, node in enumerate(nodes)}
reverse_mapping = {v: k for k, v in node_mapping.items()}  # 反向映射

# 调整进程数以匹配节点数
if size != len(nodes):
    if rank == 0:
        print("错误：MPI 进程的数量必须等于图中节点的数量。")
    MPI.Finalize()
    exit()

# 转换后的邻接表
graph = {
    node_mapping[node]: [node_mapping[neighbor] for neighbor in neighbors]
    for node, neighbors in original_graph.items()
}

# 初始化粒子数：如果 rank 对应一个节点，则设置为节点编号
particles = reverse_mapping.get(rank, 0)

# 模拟参数
steps = 1000  # 随机游走的时间步数

# 开始随机游走
for step in range(steps):
    # 为所有进程准备发送数据，即使不在图中
    send_data = [[] for _ in range(size)]  # 为每个进程准备发送列表

    if rank in graph:
        # 每个节点将粒子随机发送到相邻节点
        if particles > 0:
            for _ in range(particles):
                new_node = random.choice(graph[rank])  # 随机选择一个相邻节点
                send_data[new_node].append(1)  # 将粒子添加到对应进程的发送列表
            particles = 0  # 当前节点粒子数清零

    # 准备发送和接收计数
    send_counts = np.array([len(data) for data in send_data], dtype='i')
    recv_counts = np.array(comm.alltoall(send_counts), dtype='i')

    # 准备发送和接收缓冲区为 NumPy 数组
    send_buffer = np.array([item for sublist in send_data for item in sublist], dtype='i')
    recv_buffer = np.empty(sum(recv_counts), dtype='i')

    # 计算发送和接收的位移
    send_displs = np.array([0] + np.cumsum(send_counts[:-1]).tolist(), dtype='i')
    recv_displs = np.array([0] + np.cumsum(recv_counts[:-1]).tolist(), dtype='i')

    # 执行 Alltoallv 通信
    comm.Alltoallv([send_buffer, send_counts, send_displs, MPI.INT],
                   [recv_buffer, recv_counts, recv_displs, MPI.INT])

    # 更新粒子数量
    particles += np.sum(recv_buffer)

    # 同步所有节点
    comm.Barrier()

# 汇总最终粒子数
total_particles = comm.reduce(particles, op=MPI.SUM, root=0)

# 输出每个节点的最终粒子分布
if rank in reverse_mapping:
    print(f"节点 {reverse_mapping[rank]} 最终有 {particles} 个粒子")

# 仅由 root 节点输出统计信息
if rank == 0:
    print(f"最终的总粒子数：{total_particles}")