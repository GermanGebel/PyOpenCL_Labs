# PyOpenCL_Labs

## The first lab is a matr multiplication as [M x K] X [K x N] = [M x N]

!python Lab1\lab1.py matr_mul_kernels.cl 1024 512 2048 check

## The second lab is a transpose matr ([M x N] -> [N x M]) which devided for 3 solutions

### - naive transpose
### - with using local memory
### - with using local memory and without bank conflicts

!python Lab2\lab2.py transpose_kernels.cl 1024 512 check
=======
# PyOpenCL_Labs

## The first lab is a matr multiplication as [M x K] X [K x N] = [M x N]

### Run code:

```
python Lab1\lab1.py matr_mul_kernels.cl 1024 512 2048 check
```

***python Lab1\lab1.py matr_mul_kernels.cl 1024 512 2048 check***

## The second lab is a transpose matr ([M x N] -> [N x M]) which devided for 3 solutions

* naive transpose
* with using local memory
* with using local memory and without bank conflicts

### Run code:
```
python Lab2\lab2.py transpose_kernels.cl 1024 512 check
```


## More information
* [Instaling PyOpenCL for Windows](https://wiki.tiker.net/PyOpenCL/Installation/Windows/#installing-pyopencl-on-windows)
* [OpenCL Concepts](https://sites.google.com/site/csc8820/opencl-basics/opencl-concepts)
* [Nvidia OpenCL Best Practice Guide](https://www.nvidia.com/content/cudazone/CUDABrowser/downloads/papers/NVIDIA_OpenCL_BestPracticesGuide.pdf)
* [Nvidia OpenCL Programmig Guide](http://developer.download.nvidia.com/compute/DevZone/docs/html/OpenCL/doc/OpenCL_Programming_Guide.pdf)
