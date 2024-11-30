import threading

counter = 0

def increment():
    global counter
    for _ in range(1000000):
        counter += 1

# 创建两个线程
thread1 = threading.Thread(target=increment)
thread2 = threading.Thread(target=increment)

# 启动线程
thread1.start()
thread2.start()
# 等待线程执行完成

thread1.join()

thread2.join()

print(f"Final counter value: {counter}")