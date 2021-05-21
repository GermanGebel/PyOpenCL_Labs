__kernel void global_matr_mul(
    __global int* A, 
    __global int* B, 
    __global int* C, int M, int K, int N)
{
    int col = get_global_id(0); // N
    int row = get_global_id(1); // M

    if (row < M && col < N)
    {
        int sum = 0;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}




