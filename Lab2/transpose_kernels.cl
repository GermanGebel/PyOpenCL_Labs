__kernel void global_matr_T(
    __global int* A, 
    __global int* A_T, 
    int M, int N) {
    int i = get_global_id(0); // N
    int j = get_global_id(1); // M

    if (j < M && i < N)
        A_T[i * M + j] = A[j * N + i];
    // A_T[i * N + j] = A[j * M + i];
};

#define TILE_SIZE %(tile_size)d

__kernel void local_matr_T(
    __global int* A, 
    __global int* A_T, 
    int M, int N){
    
    __local int tile[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < N && j < M){
        tile[local_j][local_i] = A[j * N + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        int new_i = i - local_i + local_j;
        int new_j = j - local_j + local_i;
        A_T[new_i * M + new_j] = tile[local_i][local_j];
    }  
}

__kernel void padding_local_matr_T(
    __global int* A, 
    __global int* A_T, 
    int M, int N){

    __local int tile[TILE_SIZE][TILE_SIZE + 1];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int i = get_global_id(0); // for matr M x N
    int j = get_global_id(1);

    if (i < N && j < M){
        tile[local_j][local_i] = A[j * N + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        int new_i = i - local_i + local_j; // for matr N x M
        int new_j = j - local_j + local_i;
        A_T[new_i * M + new_j] = tile[local_i][local_j];
    }  
}





