__kernel void global_matr_T(
    __global int* A, 
    __global int* A_T, 
    int M, int N) {
    int col = get_global_id(0); // N
    int row = get_global_id(1); // M

    if (row < M && col < N)
        A_T[col * M + row] = A[row * N + col];
    // A_T[col * N + row] = A[row * M + col];
};

#define TILE_SIZE %(tile_size)d

__kernel void local_matr_T(
    __global int* A, 
    __global int* A_T, 
    int M, int N){
    
    __local int tile[TILE_SIZE][TILE_SIZE];

    int l_col = get_local_id(0);
    int l_row = get_local_id(1);

    int col = get_group_id(0) * TILE_SIZE + l_col;
    int row = get_group_id(1) * TILE_SIZE + l_row;

    if (col < N && row < M){
        tile[l_row][l_col] = A[row * N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        A_T[col * M + row] = tile[l_row][l_col];
    }  
}

__kernel void padding_local_matr_T(
    __global int* A, 
    __global int* A_T, 
    int M, int N){

    __local int tile[TILE_SIZE][TILE_SIZE + 1];

    int l_col = get_local_id(0);
    int l_row = get_local_id(1);

    int col = get_group_id(0) * TILE_SIZE + l_col;
    int row = get_group_id(1) * TILE_SIZE + l_row;

    if (col < N && row < M){
        tile[l_row][l_col] = A[row * N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        A_T[row * N + col] = tile[l_col][l_row];
    }
}





