#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "symgs_cpu.h"
#include "symgs_gpu.h"

d

int main(int argc, const char *argv[]) {
    // include your main function here
    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    float *matrixDiagonal;

    const char *filename = argv[1];

	// Reading a sparse matrix
	// Compressing the matrix using the CSR format (Compressed Sparse Row)
    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    float *x = (float *)malloc(num_rows * sizeof(float));

	//To fix read_matrix building of the col_ind array
    col_ind[num_vals-1] = num_rows - 1; 
    
    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
    }
    
    // From now on implementation o GPU Algorithm:
    // Allocate device memory
    int *D_row_ptr, *D_col_ind, *D_done, *D_loop;
    float *D_values, *D_matrixDiag;
    float *D_x, *D_y;
    // forward sweep: x used as x_old, y used as x_new
    // backward sweep: y used as x_old, x used as x_new
    int loop = 1; //Used to set while later
    
    
    // Memory Allocation for Kernel variables
    CHECK(cudaMalloc((void**)&D_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&D_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc((void**)&D_done, num_rows * sizeof(int)));
    CHECK(cudaMalloc((void**)&D_loop, sizeof(int)));
    CHECK(cudaMalloc((void**)&D_values, num_vals * sizeof(float)));
    CHECK(cudaMalloc((void**)&D_x, num_rows * sizeof(float)));
    CHECK(cudaMalloc((void**)&D_y, num_rows * sizeof(float)));
    CHECK(cudaMalloc((void**)&D_matrixDiag, num_rows * sizeof(float)));

    //M emory copy from Host to GPU memory
    CHECK(cudaMemcpy(D_row_ptr, row_ptr,  (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D_col_ind, col_ind,  num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D_loop, &loop, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(D_done, 0, num_rows * sizeof(int)));
    CHECK(cudaMemcpy(D_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D_matrixDiag, matrixDiagonal, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // SYMGS computing by GPU
    dim3 blocksPerGrid(ceil(num_rows/1024)+1);
    dim3 threadsPerBlock(1024);
    
    double start_gpu, end_gpu;
    start_gpu = get_time();
    
    // Forward Sweep
    loop = 1; //resetting loop variable
    while(loop == 1)
    {
    	CHECK(cudaMemset(D_loop, 0, sizeof(int)));
    	fsweep<<<blocksPerGrid,threadsPerBlock>>>(D_row_ptr, D_col_ind, D_values, num_rows, D_x, D_y, D_done, D_matrixDiag, D_loop);
    	CHECK_KERNELCALL();
    	cudaDeviceSynchronize();
    	CHECK(cudaMemcpy(&loop, D_loop, sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    //B ackward Sweep (Our matrix is Lower Triangular dominant so it only does a cycle and leave)
    loop = 1; //resetting loop variable
    while(loop == 1)
    {	
    	CHECK(cudaMemset(D_loop, 0, sizeof(int)));
    	bsweep<<<blocksPerGrid,threadsPerBlock>>>(D_row_ptr, D_col_ind, D_values, num_rows, D_x, D_y, D_done, D_matrixDiag, D_loop);
    	CHECK_KERNELCALL();
    	cudaDeviceSynchronize();
    	CHECK(cudaMemcpy(&loop, D_loop, sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    
    end_gpu = get_time();

    // Write back the results
    float *x_hw = (float*) malloc(num_rows * sizeof(float));
    CHECK(cudaMemcpy(x_hw, D_x, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
        
    // SYMGS computing by CPU
    double start_cpu, end_cpu;
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();
    
    // Print time
    printf("\nSYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("\nSYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);

    // Free main memory
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x_hw);
    free(x);
    
	// Free and Reset resources used by GPU in this process
    CHECK(cudaDeviceReset());

    return 0;
}
