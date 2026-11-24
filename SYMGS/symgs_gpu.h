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

// Forward sweep: rows from first to last.
__global__ void fsweep(	const int*  __restrict__ row_ptr, 
						const int*  __restrict__ col_ind,
						const float* __restrict__ values,
						const int num_rows,
						const float*  __restrict__ x,
						float *y,
						volatile bool *done,
						const float* __restrict__ matrixDiag,
						int *loop,
						const int pass)   // id of this launch, written to *loop if a row gives up
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= num_rows || done[tid]) return; //Execute only on the rows not computed until now
	
	float sum = x[tid];
	// Computing until column=row_end or dependencies with non-computed column are found
	for(int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++)
	{
		const int c = col_ind[i];
		float v;
		if(c >= tid)
		{
			v = x[c];
		}
		else
		{
			if (!done[c]) {
				*loop = pass; // host compares with the last pass id, so no memset is needed
				return;
			}
			__threadfence();
			
			v = __ldcg(&y[c]);
		}
		sum -= values[i] * v;
	}
	sum += x[tid] * matrixDiag[tid];
	y[tid] = sum / matrixDiag[tid];
	__threadfence();
	done[tid] = true;
}

// Backward sweep: rows from last to first.
__global__ void bsweep(const int*   __restrict__ row_ptr,
                       const int*   __restrict__ col_ind,
                       const float* __restrict__ values,
                       const int num_rows,
                       float* x,
                       const float* __restrict__ y,
                       volatile bool* done,
                       const float* __restrict__ matrixDiag,
                       int* loop,
                       const int pass)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_rows || !done[tid]) return;

	float sum = y[tid];
	for (int i = row_ptr[tid]; i < row_ptr[tid + 1]; i++)
	{
		const int c = col_ind[i];
		float v;
		if (c <= tid)
		{
			v = y[c];                 
		}
		else
		{
			if (done[c]) {
				*loop = pass;
				return;
			}
			__threadfence();        
			v = __ldcg(&x[c]);       
		}
		sum -= values[i] * v;
	}
	sum += y[tid] * matrixDiag[tid];
	x[tid] = sum / matrixDiag[tid];
	__threadfence();                 
	done[tid] = false;
}

