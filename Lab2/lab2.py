import numpy as np
import pyopencl as cl
from pyopencl.tools import get_test_platforms_and_devices
import sys
from time import time
from pprint import pprint

cl.PYOPENCL_CTX = '1:2'

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
    context = cl.create_some_context()
    queue = cl.CommandQueue(context,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    #print("Prop: {}".format(queue.set_property(cl.command_queue_properties.PROFILING_ENABLE, True)))
    
    with open(KERNELS_FILE, 'r') as kernel_file:
        program = cl.Program(context, kernel_file.read() % {'tile_size': tile_size}).build()
        kernel_file.close()

def fill_array(M, N):
    a = []
    for i in range(M):
        for j in range(N):
            a.append(i*N+j)
    return np.array(a)

def host_transpose(matr, M, N):
    matr_T = np.empty_like(matr)
    for i in range(M):
        for j in range(N):
            matr_T[j * M + i] = matr[i * N + j]
    return matr_T


def lab2(M, N, check_results):

    global context, queue, program, tile_size

    does_matr_print = 0

    print(f"LAB 2\n[{M}x{N}] -> [{N}x{M}]\n")
    results = {}
    kernel_names = ['global_matr_T', 'local_matr_T', '_local_matr_T']
    # kernels = program.all_kernels()

    matr = fill_array(M, N)
    
    if does_matr_print:
        print(f"Matr: \n {[i for i in matr]}\n")
    #print(matr.nbytes)
    host_matr_T, device_matr_T = None, np.empty_like(matr)
    cpu_time = None

    if check_results:
        start = time()
        host_matr_T = host_transpose(matr, M, N)
        if does_matr_print:
            print(f"Host matr_T: \n {[i for i in host_matr_T]}")
        cpu_time = time() - start
    
    buffer_in = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matr)
    buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matr.nbytes)
    
    for i in kernel_names:
        kernel = None  
        if i == 'global_matr_T' :
            kernel = program.global_matr_T
        elif i == 'local_matr_T':
            kernel = program._local_matr_T
        elif i == '_local_matr_T':
            kernel = program.local_matr_T
        kernel.set_args(buffer_in, buffer_out, M, N)
        check_amount = 50
        gpu_time = 0
        for j in range(check_amount):
            event = cl.enqueue_nd_range_kernel(queue, kernel, (M, N), None)
            event.wait()
            gpu_time += (event.profile.end  - event.profile.start)
        gpu_time /= check_amount
        cl.enqueue_copy(queue, device_matr_T, buffer_out)
        data = {}
        if does_matr_print:
            print(f"Device matr_T: \n {[ i for i in device_matr_T]}")
        if check_results:            
            #print('CPU_TIME: {}s\nGPU_TIME: {}ms'.format(cpu_time, gpu_time))
            flag = True
            for j in range(len(host_matr_T)):
                if host_matr_T[j] != device_matr_T[j]:
                    flag = False
                    print(j)
                    break
            data.update({"CPU_TIME (s)": cpu_time, 'Compare with host': flag})
        
        data.update({"GPU_TIME (ms)": gpu_time/1e6})

        mem_bw = (2*matr.nbytes/(gpu_time*1e-9))/pow(1024, 3)
        efficiency = (mem_bw / 64 * 100)
        data.update({'Mem bandwidths (GB/s)': mem_bw, 'Efficiency(%)': efficiency})

        results.update({i: data})
    return results

def graphics():
    
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
    
    with open('kernels.cl', 'r') as kernel_file:
        program = cl.Program(context, kernel_file.read() % {'tile_size': tile_size}).build()
        kernel_file.close()
    sizes = [int(((2**i) // 32) * 32)
            for i in np.arange(10, 13, 0.125)]

    efficiecies = {}
    
    data = lab2(np.int32(100), np.int32(100), False)
    for name in data.keys():
        efficiecies[name] = []

    for size in sizes:
        data = lab2(np.int32(size), np.int32(size), False)
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

    savefig("transpose-benchmarkkk.png")
    
        
def lab2_parse_argv(argv):
    global KERNELS_FILE
    if len(argv) == 1:
        print("Enter more arguments")
        return
    elif len(argv) == 4:        
        check_results = False
    elif len(argv) == 5:
        check_results = True if argv[4] == 'check' else False
    else:
        print("Enter less arguments")
        return
    KERNELS_FILE = argv[1]
    M = np.int32(argv[2])
    N = np.int32(argv[3])
    initOpenCL()
    pprint(lab2(M, N, check_results))


def main():   
    lab2_parse_argv(sys.argv)
    #graphics()

if __name__ == '__main__':
    main()
    