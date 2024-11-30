from mpi4py import MPI
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
particles = 1 if rank in graph else 0

# 模拟参数
steps = 1000  # 随机游走的时间步数

for step in range(steps):
    # 非阻塞发送粒子到目标节点
    send_reqs = []
    if rank in graph and particles > 0:
        for _ in range(particles):
            target = random.choice(graph[rank])
            send_reqs.append(comm.isend(1, dest=target))  # 异步发送粒子
        particles = 0  # 当前节点粒子清零

    # 非阻塞接收来自其他节点的粒子
    incoming_particles = 0
    recv_reqs = []
    for neighbor in range(size):
        if neighbor != rank:  # 避免自己接收自己的粒子
            req = comm.irecv(source=neighbor)
            recv_reqs.append(req)

    # 等待所有接收完成
    for req in recv_reqs:
        try:
            incoming_particles += req.wait()  # 获取接收到的粒子数
        except MPI.Exception:
            pass  # 如果没有消息则忽略

    # 确保所有消息发送完成
    MPI.Request.Waitall(send_reqs)

    # 更新当前粒子数
    particles += incoming_particles

    # 同步所有节点，确保每个时间步结束后同步
    comm.Barrier()

# 汇总最终粒子数
total_particles = comm.reduce(particles, op=MPI.SUM, root=0)

# 输出每个节点的最终粒子分布
if rank in reverse_mapping:
    print(f"节点 {reverse_mapping[rank]} 最终有 {particles} 个粒子")

# 仅由 root 节点输出统计信息
if rank == 0:
    print(f"最终的总粒子数：{total_particles}")