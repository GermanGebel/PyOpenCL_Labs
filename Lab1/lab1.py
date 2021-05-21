from os import cpu_count
import numpy as np
import pyopencl as cl
from pyopencl.tools import get_test_platforms_and_devices
import sys
from time import time
from pprint import pprint

KERNELS_FILE: str
context: cl.Context
queue: cl.CommandQueue
program: None  
tile_size = 64


def initOpenCL():
    global context, queue, program
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(cl.device_type.GPU)
    device = devices[0]
    global_mem = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
    local_mem = device.get_info(cl.device_info.LOCAL_MEM_SIZE)
    max_CU = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
    max_WGS = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    info = """Device: {}:
Global mem size: {} Bytes
Local mem size: {} Bytes
Max CU: {}
Max work groub size: {} WIs\n""".format( device.name, global_mem, local_mem, max_CU, max_WGS)
    print(info)
    context = cl.Context(devices)#cl.create_some_context()
    queue = cl.CommandQueue(context,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    #print("Prop: {}".format(queue.set_property(cl.command_queue_properties.PROFILING_ENABLE, True)))
    
    with open(KERNELS_FILE, 'r') as kernel_file:
        program = cl.Program(context, kernel_file.read() % {'tile_size': tile_size}).build()
        kernel_file.close()

def host_mul(A, B):
    sum = np.int32(0)

    start = time()
    C = A @ B
    cpu_time = time() - start

    return C, cpu_time

def global_normilize(G, L):
    while not G % L == 0:
        G += 1
    return G

def normilize_sizes(M, N):
    local_size_M = 1
    i = 0
    barrier = 32
    while(local_size_M <= M and local_size_M <= barrier and 2 ** (i + 1) <= barrier and M > 2 ** (i + 1)):
        # print(1, local_size_M <= M, local_size_M <= barrier, M > 2 ** (i + 1))
        i += 1
        local_size_M = 2 ** i
    
    # print(local_size_M)
        
    M = global_normilize(M, local_size_M)
    # print(M)

    local_size_N = 1
    barrier = 1024 // local_size_M
    # print(barrier)
    i = 0
    
    while(local_size_N <= N and local_size_N <= barrier and 2 ** (i + 1) <= barrier and N > 2 ** (i + 1)):
        # print(2, local_size_N <= M, local_size_N <= barrier, N > 2 ** (i + 1))
        i += 1
        local_size_N = 2 ** i

    # print(local_size_N)
    N = global_normilize(N, local_size_N)
    # print(N)

    return M, N, local_size_M, local_size_N



def lab1(M, K, N, check_results, print_results):
    global context, queue, program, tile_size

    print(f"LAB 1\n[{M}x{K}] X [{K}x{N}]\n")

    # kernels = program.all_kernels()
    low = 100
    high = 1000

    A = np.random.randint(low, high, M * K).astype(np.int32).reshape(M, K)
    B = np.random.randint(low, high, K * N).astype(np.int32).reshape(K, N)
    
    if print_results:
        print(f"Matr A: \n {A}\n")
        print(f"Matr B: \n {B}\n")
    # print(matr.nbytes)
    host_result_matr, device_result_matr = None, np.empty((M * N)).astype(np.int32)
    cpu_time = 0

    if check_results:
        host_result_matr, cpu_time = host_mul(A, B)
        if print_results:
            print(f"Host result matr: \n {host_result_matr}")
        
    
    buffer_in_A = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    buffer_in_B = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    buffer_out_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, device_result_matr.nbytes)
    
    # kernel_name = 'global_matr_mul'
    kernel = program.global_matr_mul

    kernel.set_args(buffer_in_A, buffer_in_B, buffer_out_C, M, K, N)
    
    # check_amount = 50
    gpu_time = 0
    
    m, n, l_m, l_n = normilize_sizes(M, N)
    global_size = (n, m)
    local_size = (l_n, l_m)
    
    print("Global size: {}\nLocal size: {}".format(global_size, local_size))
    # for j in range(check_amount):

    event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    event.wait()
    gpu_time += (event.profile.end  - event.profile.start)
    # gpu_time /= check_amount
    
    cl.enqueue_copy(queue, device_result_matr, buffer_out_C)
    device_result_matr = device_result_matr.reshape(M, N)

    if print_results:
        print(f"Device result matr: \n {device_result_matr}")
    if check_results:            
        comparison = host_result_matr == device_result_matr
        is_equal = comparison.all()
        print(f'Compare with host matr: {is_equal}')
        print('CPU_TIME: {} ms'.format(cpu_time * 1e3))
        # data.update({"CPU_TIME (s)": cpu_time, 'Compare with host': flag})
    
    # data.update({"GPU_TIME (ms)": gpu_time/1e6})
    print('GPU_TIME: {} ms'.format(gpu_time * 1e-6))

def main():   
    global KERNELS_FILE

    argv = sys.argv

    check_results = False
    print_results = False

    if len(argv) == 1:
        print("Enter more arguments")
        return
    elif len(argv) == 5:
        pass
    elif len(argv) == 6:
        check_results = True if argv[5] == 'check' else False
    elif len(argv) == 7:
        check_results = True if argv[5] == 'check' else False
        print_results = True if argv[6] == 'print' else False
    else:
        print("Enter less arguments")
        return
    KERNELS_FILE = argv[1]
    M = np.int32(argv[2])
    K = np.int32(argv[3])
    N = np.int32(argv[4])

    # if (not (M % 32 == 0 and N % 32 == 0 ))

    initOpenCL()
    start = time()
    lab1(M, K, N, check_results, print_results)
    print("Program work time:{}".format(time() - start))

if __name__ == '__main__':
    main()
    