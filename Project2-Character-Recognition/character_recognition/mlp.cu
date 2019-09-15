#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

#define blockSize 128

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
        
    // TODO: __global__

    /**
        * Example of use case (follow how you did it in stream compaction)
        */
    /*void scan(int n, int *odata, const int *idata) {
        timer().startGpuTimer();
        // TODO
        timer().endGpuTimer();
    }
    */
	__global__ void kernMatrixMultiply(float *A, float *B, float *C, int m, int n, int k) {
		/*
			A -> m X n and B is n X k
			C -> m X k the output array
		*/

		int row = (blockDim.y * blockIdx.y) + threadIdx.y;
		int col = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (row >= m || col >= k) return;

		float sum = 0.0;
		for (int i = 0; i < n; i++) {
			sum += A[row*n + i] * B[i*k + col];
		}
		C[row*k + col] = sum;
	}


	void matrixMultiply(float *A, float *B, float *C, int m, int n, int k) {

		float *dev_A, *dev_B, *dev_C;
		cudaMalloc((void **)&dev_A, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_A failed!");
		cudaMalloc((void **)&dev_B, n * k * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_B failed!");
		cudaMalloc((void **)&dev_C, m * k * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

		cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (k + blockSize - 1) / blockSize;

		dim3 dimGrid(gridRows, gridCols);
		dim3 dimBlock(blockSize, blockSize);

		kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_C, m, n, k);

		cudaMemcpy(C, dev_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_A);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
		cudaFree(dev_B);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
		cudaFree(dev_C);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
	}





	void printArray2D(float *X, int nR, int nC) {
		for (int i = 0; i < nR; i++) {
			for (int j = 0; j < nC; j++)
				printf("%.2f ", X[i*nC + j]);
			printf("\n");
		}
	}

	// TODO: implement required elements for MLP sections 1 and 2 here

	void forwardPass(int N, int d, int C, int h1, float *X, int *y,  float *loss, float *W1, float *W2) {

		float *X2 = new float[N * h1 * sizeof(float)];
		matrixMultiply(X, W1, X2, N, d, h1);

		printArray2D(X2, N, h1);

	}


}
