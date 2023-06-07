//CHECK Calls for CUDA functions
//Used to check if the calls encountered problems

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }
// CUDA kernels implemented for SYMGS algorithm
// Symmetric Gauss Seidl Algorithm works 

// Forward Sweep (working on j>i elements, where j is the column and i is the row)
// tid indicading the row and the thread identifier
// i indicating the column
__global__ void fsweep(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *y, int *done, float *matrixDiag, int *loop)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if((tid < num_rows) && (done[tid] == 0)) //Execute only on the rows not computed until now
	{
		float sum = x[tid];
		const int row_start = row_ptr[tid]; 
		const int row_end = row_ptr[tid+1];
		bool process = true;
		// Computing until column=row_end or dependencies with non-computed column are found
		for(int i = row_start; i < row_end && process; i++)
		{
			if(col_ind[i]>=tid)
			{
				sum -= values[i] * x[col_ind[i]];
			}
			else if(done[col_ind[i]] == 1)
			{
				sum -= values[i] * y[col_ind[i]];
			}
			else process = false;
		}
			if(process) // we save the new computed value in y
			{
				sum += x[tid] * matrixDiag[tid];
				y[tid] = sum / matrixDiag[tid];
				done[tid]=1;
			}
			else loop[0]=1; // if dependencies are found we do another loop cycle
	}
	__syncthreads();
}


//B ackward Sweep (working on j<i elements, where j is the column and i is the row)
// tid indicading the row and the thread identifier
// i indicating the column
__global__ void bsweep(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *y, int *done, float *matrixDiag, int *loop)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if((tid < num_rows) && (done[tid] == 1))
	{
		float sum = y[tid];
		const int row_start = row_ptr[tid];
		const int row_end = row_ptr[tid+1];
		bool process = true;
		//Computing until column=row_end or dependencies with non-computed column are found
		for(int i = row_start; (i < row_end) && process; i++)
		{
			if(col_ind[i]<=tid)
			{
				sum -= values[i] * y[col_ind[i]];
			}
			else if(done[col_ind[i]] == 0)
			{
				sum -= values[i] * x[col_ind[i]];
			}
			else process = false;
		}
			if(process) // we save the new computed value in x
			{
				sum += y[tid] * matrixDiag[tid];
				x[tid] = sum / matrixDiag[tid];
				done[tid]=0;
			}
			else loop[0]=1; // if dependencies are found we do another loop cycle
		
	}
	__syncthreads();
}

