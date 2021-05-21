# Lab 1 [M x K] X [K x N]

### Run code

```
py lab1.py matr_mul_kernels.cl 3096 1024 5120 check print
```

### Results

```
Device: NVIDIA GeForce 840M:
Global mem size: 2147483648 Bytes
Local mem size: 49152 Bytes
Max CU: 3
Max work groub size: 1024 WIs


# Result 1
LAB 1
[2x1000] X [1000x500]

Global size: (512, 2)
Local size: (256, 1)

Compare with host matr: True
CPU_TIME: 2.9976367950439453 ms
GPU_TIME: 0.21254399999999998 ms

Program work time:0.017995119094848633

# Result 2
LAB 1
[1000x300] X [300x99]

Global size: (128, 1024)
Local size: (32, 32)

GPU_TIME: 1.91088 ms
Program work time:0.01599717140197754

# Result 3
LAB 1
[3x4] X [4x5]

Matr A:
 [[508 639 958 739]
 [689 606 822 343]
 [430 106 372 693]]

Matr B:
 [[921 942 862 506 558]
 [404 515 187 539 704]
 [523 150 182 136 566]
 [840 648 728 220 521]]

Host result matr:
 [[1847818 1430193 1269737  894337 1660567]
 [1597419 1306692 1106548  862520 1455041]
 [1215530  964514  962690  477766  886169]]

Global size: (8, 4)
Local size: (4, 2)

Device result matr:
 [[1847818 1430193 1269737  894337 1660567]
 [1597419 1306692 1106548  862520 1455041]
 [1215530  964514  962690  477766  886169]]

Compare with host matr: True
CPU_TIME: 0.0 ms
GPU_TIME: 0.013056 ms

Program work time:0.012976884841918945
```

