__kernel void global_matr_T(__global int* A, __global int* A_T, int M, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    A_T[col * M + row] = A[row * N + col];
    // A_T[col * N + row] = A[row * M + col];
};

#define TILE_SIZE %(tile_size)d

__kernel void local_matr_T(__global int* A, __global int* A_T, int M, int N){
    
    __local int tile[TILE_SIZE][TILE_SIZE];

    int l_i = get_local_id(0);
    int l_j = get_local_id(1);

    int i = get_group_id(0) * TILE_SIZE + l_j;
    int j = get_group_id(1) * TILE_SIZE + l_i;

    if (i < M && j < N)
        tile[l_j * TILE_SIZE][l_i] = A[j * N + i];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < M && j < N)
        A_T[i * M + j] = tile[l_j * TILE_SIZE][l_i];  
}

__kernel void _local_matr_T(__global int* A, __global int* A_T, int M, int N){

    __local int tile[TILE_SIZE][TILE_SIZE + 1];

    int l_i = get_local_id(0);
    int l_j = get_local_id(1);

    int i = get_group_id(0) * TILE_SIZE + l_j;
    int j = get_group_id(1) * TILE_SIZE + l_i;

    if (i < M && j < N)
        tile[l_j * TILE_SIZE][l_i] = A[j * N + i];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < M && j < N)
        A_T[i * M + j] = tile[l_j * TILE_SIZE][l_i];  
}





