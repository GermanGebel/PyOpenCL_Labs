# PyOpenCL_Labs

## The first lab is a matr multiplication as [M x K] X [K x N] = [M x N]

!python Lab1\lab1.py matr_mul_kernels.cl 1024 512 2048 check

## The second lab is a transpose matr ([M x N] -> [N x M]) which devided for 3 solutions

### - naive transpose
### - with using local memory
### - with using local memory and without bank conflicts

!python Lab2\lab2.py transpose_kernels.cl 1024 512 check
