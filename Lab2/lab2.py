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
    context = cl.create_some_context()
    queue = cl.CommandQueue(context,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    #print("Prop: {}".format(queue.set_property(cl.command_queue_properties.PROFILING_ENABLE, True)))

def naive_host_transpose(matr, M, N):
    matr_T = np.empty_like(matr)

    start = time()

    for row in range(M):
        for col in range(N):
            matr_T[col * M + row] = matr[row * N + col]
    
    cpu_time = time() - start
    return matr_T, cpu_time

def host_transpose(matr):
    start = time()
    matr_T = matr.T
    end = time()
    cpu_time = end - start
    return matr_T, cpu_time

def global_normilize(G, L):
    while not G % L == 0:
        G += 1
    return G

def normilize_sizes(M, N):
    local_size_M = 1
    i = 0
    barrier = 32
    while(local_size_M <= M and local_size_M <= barrier and 2 ** (i + 1) <= barrier and M > 2 ** (i + 1)):
        i += 1
        local_size_M = 2 ** i

    M = global_normilize(M, local_size_M)

    local_size_N = 1
    barrier = local_size_M
    i = 0    
    while(local_size_N <= N and local_size_N <= barrier and 2 ** (i + 1) <= barrier and N > 2 ** (i + 1)):
        i += 1
        local_size_N = 2 ** i

    N = global_normilize(N, local_size_N)

    if local_size_M > local_size_N:
        local_size_M = local_size_N
        M = global_normilize(M, local_size_M)

    return M, N, local_size_M, local_size_N

def lab2(M, N, check_results, print_results):
    global context, queue, program, KERNELS_FILE

    m, n, l_m, l_n = normilize_sizes(M, N)

    global_size = (n, m)
    local_size = (l_n, l_m)

    with open(KERNELS_FILE, 'r') as kernel_file:
        program = cl.Program(context, kernel_file.read() % {'tile_size': l_n}).build()
        kernel_file.close()

    print(f"LAB 2\n[{M}x{N}] -> [{N}x{M}]\n")
    results = {}
    kernel_names = ['global_matr_T', 'local_matr_T', 'padding_local_matr_T']
    # kernels = program.all_kernels()

    low = 10
    high = 50

    matr = np.random.randint(low, high, M * N).astype(np.int32)
    
    if print_results:
        print("Matr: \n {}\n".format(matr.reshape((M, N))))
    #print(matr.nbytes)
    host_matr_T, device_matr_T = None, np.empty_like(matr)
    cpu_time = None

    if check_results:
        host_matr_T, cpu_time = host_transpose(matr.reshape((M, N))) #naive_host_transpose(matr, M, N)
        if print_results:
            print("Host matr_T: \n {}\n".format(host_matr_T))
        
    
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matr)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matr.nbytes)

    print("Global size: {}\nLocal size: {}\n".format(global_size, local_size))
    
    for i in kernel_names:
        kernel = None  
        if i == 'global_matr_T' :
            kernel = program.global_matr_T
            # local_size = None
        elif i == 'local_matr_T':
            kernel = program.local_matr_T
            local_size = None
        elif i == 'padding_local_matr_T':
            kernel = program.padding_local_matr_T
            local_size = None
        kernel.set_args(buffer_in, buffer_out, M, N)

        check_amount = 50
        gpu_time = 0
        for j in range(check_amount):
            event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
            event.wait()
            gpu_time += (event.profile.end  - event.profile.start)
        gpu_time /= check_amount

        cl.enqueue_copy(queue, device_matr_T, buffer_out)
        device_matr_T.shape = (N, M)

        data = {}
        if print_results:
            print("Device matr_T: \n {}".format(device_matr_T))
        if check_results:            
            #print('CPU_TIME: {}s\nGPU_TIME: {}ms'.format(cpu_time, gpu_time))
            comparison = host_matr_T == device_matr_T
            is_equal = comparison.all()
            data.update({"CPU_TIME (s)": cpu_time, 'Compare with host': is_equal})
        
        data.update({"GPU_TIME (ms)": gpu_time * 1e-6})

        mem_bw = (2 * device_matr_T.nbytes / pow(1024, 3)) / (gpu_time * 1e-6) # GB / s 
        efficiency = (mem_bw  / 80 * 100)
        data.update({'Mem bandwidths (GB/s)': mem_bw, 'Efficiency(%)': efficiency})

        results.update({i: data})
    return results

def new_name():
    import os
    files = os.listdir('bench')
    return 'bench/test_{}.png'.format(len(files))


def graphics():    
    global context, queue, program, KERNELS_FILE
    KERNELS_FILE = 'transpose_kernels.cl'

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
    context = cl.create_some_context()
    queue = cl.CommandQueue(context,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    #print("Prop: {}".format(queue.set_property(cl.command_queue_properties.PROFILING_ENABLE, True)))
    
    sizes = [int(((2**i) // 32) * 32)
            for i in np.arange(10, 13.5, 0.125)]

    efficiecies = {}
    
    data = lab2(np.int32(100), np.int32(100), False, False)
    for name in data.keys():
        efficiecies[name] = []

    for size in sizes:
        data = lab2(np.int32(size), np.int32(size), False, False)
        for name in data.keys():
            efficiecies[name].append(data[name]['Efficiency(%)'])

    from matplotlib.pyplot import clf, plot, title, xlabel, ylabel, \
                savefig, legend, grid
    for name in efficiecies.keys():
        plot(sizes, np.array(efficiecies[name]), "o-", label=name)

    xlabel("Matrix NDRange")
    ylabel("Efficiency(%)")
    legend(loc="best")
    grid()

    savefig(new_name())
    
        
def lab2_parse_argv(argv):
    global KERNELS_FILE

    check_results = False
    print_results = False

    if len(argv) == 1:
        print("Enter more arguments")
        return
    elif len(argv) == 5:
        check_results = True if argv[4] == 'check' else False
    elif len(argv) == 6:
        check_results = True if argv[4] == 'check' else False
        print_results = True if argv[5] == 'print' else False
    else:
        print("Enter less arguments")
        return
    KERNELS_FILE = argv[1]
    M = np.int32(argv[2])
    N = np.int32(argv[3])
    initOpenCL()
    pprint(lab2(M, N, check_results, print_results))


def main():   
    lab2_parse_argv(sys.argv)
    # graphics()

if __name__ == '__main__':
    main()
    