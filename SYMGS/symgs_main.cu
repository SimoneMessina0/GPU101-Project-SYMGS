#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "symgs_cpu.h"
#include "symgs_gpu.h"

int main(int argc, const char *argv[]) {
    // include your main function here
    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file\n");
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

	//The file has one entry less than its header: drop the empty slot at the end
    if (col_ind[num_vals-1] == -1)
    {
        num_vals--;
        row_ptr[num_rows] = num_vals;
    }

    // Generate a random vector (float division, so the values are not mostly 0)
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (float)(rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
    }

    // From now on implementation o GPU Algorithm:
    // Allocate device memory
    int *D_row_ptr, *D_col_ind, *D_loop;
    bool *D_done;
    float *D_values, *D_matrixDiag;
    float *D_x, *D_y;
    // forward sweep: x used as x_old, y used as x_new
    // backward sweep: y used as x_old, x used as x_new

    // Memory Allocation for Kernel variables
    CHECK(cudaMalloc((void**)&D_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&D_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc((void**)&D_done, num_rows * sizeof(bool)));
    CHECK(cudaMalloc((void**)&D_loop, sizeof(int)));
    CHECK(cudaMalloc((void**)&D_values, num_vals * sizeof(float)));
    CHECK(cudaMalloc((void**)&D_x, num_rows * sizeof(float)));
    CHECK(cudaMalloc((void**)&D_y, num_rows * sizeof(float)));
    CHECK(cudaMalloc((void**)&D_matrixDiag, num_rows * sizeof(float)));

    // Memory copy from Host to GPU memory
    CHECK(cudaMemcpy(D_row_ptr, row_ptr,  (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D_col_ind, col_ind,  num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(D_loop, 0, sizeof(int)));          // pass ids start at 1, so 0 never matches
    CHECK(cudaMemset(D_done, 0, num_rows * sizeof(bool)));
    CHECK(cudaMemcpy(D_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D_matrixDiag, matrixDiagonal, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // SYMGS computing by GPU
    const int TPB = 256;
    dim3 threadsPerBlock(TPB);
    dim3 blocksPerGrid((num_rows + TPB - 1) / TPB);   // exact, no surplus block

    // Pinned host flag: makes each 4-byte readback cheaper
    int *h_loop;
    CHECK(cudaMallocHost((void**)&h_loop, sizeof(int)));

    int pass = 0;      // unique id per launch, only ever increases (never reused)

    // Runs one sweep until every row is done; returns the number of launches.
    // A row that gives up writes the current pass id into D_loop, so the sweep
    // is finished when D_loop does not hold the id of the last launch.
    auto run_sweep = [&](bool forward) {
        int launches = 0;
        do {
            ++pass;
            if (forward)
                fsweep<<<blocksPerGrid, threadsPerBlock>>>(D_row_ptr, D_col_ind, D_values, num_rows,
                                                           D_x, D_y, D_done, D_matrixDiag, D_loop, pass);
            else
                bsweep<<<blocksPerGrid, threadsPerBlock>>>(D_row_ptr, D_col_ind, D_values, num_rows,
                                                           D_x, D_y, D_done, D_matrixDiag, D_loop, pass);
            launches++;
            CHECK_KERNELCALL();
            CHECK(cudaMemcpy(h_loop, D_loop, sizeof(int), cudaMemcpyDeviceToHost));
        } while (*h_loop == pass);
        return launches;
    };

    // ---- Warm-up (first launch pays for module loading) ----
    run_sweep(true);
    run_sweep(false);
    // After a full run done[] is all 0 again (bsweep resets it), so only x needs restoring
    CHECK(cudaMemcpy(D_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Timed run with CUDA events ----
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));
    CHECK(cudaEventRecord(ev0));
    int f_launches = run_sweep(true);    // Forward Sweep
    int b_launches = run_sweep(false);   // Backward Sweep
    CHECK(cudaEventRecord(ev1));
    CHECK(cudaEventSynchronize(ev1));
    float gpu_ms;
    CHECK(cudaEventElapsedTime(&gpu_ms, ev0, ev1));

    // Write back the results
    float *x_hw = (float*) malloc(num_rows * sizeof(float));
    CHECK(cudaMemcpy(x_hw, D_x, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // SYMGS computing by CPU
    double start_cpu, end_cpu;
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // Correctness check, relative to the largest value (nvcc fuses `sum -= a*b` into FMA, host code may not)
    double max_err = 0.0, max_val = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        max_err = fmax(max_err, fabs((double)x_hw[i] - x[i]));
        max_val = fmax(max_val, fabs((double)x[i]));
    }
    double err = max_err / max_val;
    int ok = err < 1e-4;
    printf("\nCheck: max err %.3e (relative to max |x| %.3e) -> %s\n", err, max_val, ok ? "PASS" : "FAIL");
    printf("Launches: forward %d, backward %d\n", f_launches, b_launches);

    // Print time
    printf("\nSYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("\nSYMGS Time GPU: %.10lf\n", gpu_ms * 1e-3);

    // Free main memory
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x_hw);
    free(x);
    CHECK(cudaFreeHost(h_loop));
    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));

	// Free and Reset resources used by GPU in this process
    CHECK(cudaDeviceReset());

    return ok ? 0 : 1;
}
