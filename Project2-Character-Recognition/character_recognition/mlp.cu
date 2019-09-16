#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

#define blockSize 32

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

	__global__ void kernExp(float *A, int m, int n) {
		int ty = (blockDim.y * blockIdx.y) + threadIdx.y;
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (ty >= m || tx >= n) return;

		A[ty*n + tx] = exp(A[ty*n + tx]);
	}

	__global__ void kernReLU(float *A, int m, int n) {
		int ty = (blockDim.y * blockIdx.y) + threadIdx.y;
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (ty >= m || tx >= n) return;

		if (A[ty*n + tx] < 0) A[ty*n + tx] = 0.0;
	}

	__global__ void kernDerivativeReLU(float *A, int m, int n) {
		int ty = (blockDim.y * blockIdx.y) + threadIdx.y;
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (ty >= m || tx >= n) return;

		if (A[ty*n + tx] > 0) {
			A[ty*n + tx] = 1.0;
		}
		else {
			A[ty*n + tx] = 0.0;
		}
	}

	__global__ void kernMax(float *A, float *max, int N, int C) {
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (tx >= N) return;

		int maxIndex = 0;
		float maxVal = A[tx*N + 0];
		for (int i = 1; i < C; i++) {
			float val = A[tx*N + i];
			if (val > maxVal) {
				maxVal = val;
				maxIndex = i;
			}
		}

		max[tx] = maxIndex;
	}

	__global__ void kernMatrixTranspose(float* A, float* B, int m,  int n)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		if (tx >= n || ty >= m) return;


		int idx = ty * n + tx;
		int transIdx = tx * m + ty;
		B[transIdx] = A[idx];
	
	}

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

	__global__ void kernElementMatrixMultiply(float *A, float *B, float *C, int m, int n) {
		/*
			A -> m X n and B is n X k
			C -> m X k the output array
		*/

		int ty = (blockDim.y * blockIdx.y) + threadIdx.y;
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (ty >= m || tx >= n) return;

		C[ty*n + tx] = A[ty*n + tx] * B[ty*n + tx];

	}

	__global__ void kernElementMatrixAdd(float *A, float *B, float *C, float alpha, int m, int n) {
		/*
			A -> m X n and B is n X k
			C -> m X k the output array
		*/

		int ty = (blockDim.y * blockIdx.y) + threadIdx.y;
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (ty >= m || tx >= n) return;

		C[ty*n + tx] = A[ty*n + tx] - alpha * B[ty*n + tx];

	}

	__global__ void kernDerivativeLossScores(float *probs, int *y, float *dscores, int N, int C) {
		int ty = (blockDim.y * blockIdx.y) + threadIdx.y;
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (ty >= N || tx >= C) return;

		if (tx == y[ty]) {
			dscores[ty*C + tx] = probs[ty*C + tx] - 1;
			dscores[ty*C + tx] /= N;
		} else {
			dscores[ty*C + tx] /= N;
		}

	}

	void softmaxExp(float *A, int m, int n) {
		float *dev_A;
		cudaMalloc((void **)&dev_A, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

		cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernExp << <dimGrid, dimBlock >> > (dev_A, m, n);


		cudaMemcpy(A, dev_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_A);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
	}

	void softmaxNormalize(float *A, int m, int n) {
		// TODO: Should be parallelized

		for (int i = 0; i < m; i++) {
			float sum = 0;
			for (int j = 0; j < n; j++)
				sum += A[i*n + j];

			for (int j = 0; j < n; j++)
				A[i*n + j] /= sum;
		}
	}

	void calculateLoss(float *probs, int N, int C, int *y, float* loss) {
		// TODO: Should be parallelized
		float totalLoss = 0;
		for (int i = 0; i < N; i++) {
			int label = y[i];
			totalLoss += -log(probs[i*C + label]);
		}

		totalLoss /= N;

		*loss = totalLoss;
	}




	void softmaxDerivativeWrtScores(float *probs, int *y, float *dscores, int N, int C) {
		/*
		Calcluates dL/dscores . probs = softmax(scores)
		dL/dscores = probs[range(N), y] -= 1
		*/

		float *dev_dscores, *dev_probs;
		cudaMalloc((void **)&dev_dscores, N * C * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_dscores failed!");
		cudaMalloc((void **)&dev_probs, N * C * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_probs failed!");

		int *dev_y;
		cudaMalloc((void **)&dev_y, N * sizeof(int));
		checkCUDAErrorWithLine("cudaMalloc dev_y failed!");

		cudaMemcpy(dev_dscores, probs, N * C * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_probs, probs, N * C * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (N + blockSize - 1) / blockSize;
		int gridCols = (C + blockSize - 1) / blockSize;
		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);
		kernDerivativeLossScores<<<dimGrid, dimBlock>>>(dev_probs, dev_y, dev_dscores, N, C);

		cudaMemcpy(dscores, dev_dscores, N * C * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_dscores);
		checkCUDAErrorWithLine("cudaFree dev_dscores failed!");
		cudaFree(dev_probs);
		checkCUDAErrorWithLine("cudaFree dev_probs failed!");
		cudaFree(dev_y);
		checkCUDAErrorWithLine("cudaFree dev_y failed!");
		
	}

	void reLU(float *A, int m, int n) {
		float *dev_A;
		cudaMalloc((void **)&dev_A, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

		cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernReLU<<<dimGrid, dimBlock>>>(dev_A, m, n);


		cudaMemcpy(A, dev_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_A);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
	}

	void derivativeReLU(float *A, int m, int n) {
		float *dev_A;
		cudaMalloc((void **)&dev_A, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

		cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernDerivativeReLU << <dimGrid, dimBlock >> > (dev_A, m, n);


		cudaMemcpy(A, dev_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_A);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
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

		dim3 dimGrid(gridCols, gridRows);
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

	void matrixElementMultiply(float *A, float *B, float *C, int m, int n) {

		float *dev_A, *dev_B, *dev_C;
		cudaMalloc((void **)&dev_A, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_A failed!");
		cudaMalloc((void **)&dev_B, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_B failed!");
		cudaMalloc((void **)&dev_C, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

		cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_B, B, m * n * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernElementMatrixMultiply << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_C, m, n);

		cudaMemcpy(C, dev_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_A);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
		cudaFree(dev_B);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
		cudaFree(dev_C);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
	}

	void updateWeights(float *A, float *B, float *C, float alpha, int m, int n) {

		float *dev_A, *dev_B, *dev_C;
		cudaMalloc((void **)&dev_A, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_A failed!");
		cudaMalloc((void **)&dev_B, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_B failed!");
		cudaMalloc((void **)&dev_C, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

		cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_B, B, m * n * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernElementMatrixAdd << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_C, alpha, m, n);

		cudaMemcpy(C, dev_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_A);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
		cudaFree(dev_B);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
		cudaFree(dev_C);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
	}

	void matrixTranspose(float *A, float *B, int m, int n) {

		float *dev_A, *dev_B;
		cudaMalloc((void **)&dev_A, m * n * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_A failed!");
		cudaMalloc((void **)&dev_B, n * m * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

		cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernMatrixTranspose << <dimGrid, dimBlock >> > (dev_A, dev_B, m, n);

		cudaMemcpy(B, dev_B, n * m * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_A);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
		cudaFree(dev_B);
		checkCUDAErrorWithLine("cudaFree dev_A failed!");
	}


	void calculateDerviativeW2(int N, int d, int C, int h1, float *dscores, float *X2, float *dW2) {
		float *X2Trans = new float[h1 * N * sizeof(float)];
		matrixTranspose(X2, X2Trans, N, h1);


		// dL/dW2 = X2.T * dscores (h1xN X NxC  = h1xC)
		matrixMultiply(X2Trans, dscores, dW2, h1, N, C);

		delete[] X2Trans;
	}

	void calculateDerviativeW1(int N, int d, int C, int h1, float *X, float *fc, float *W2, float *dscores, float *dW1) {
		float *dW1_1 = new float[N * h1 * sizeof(float)];
		float *W2Trans = new float[C * h1 * sizeof(float)];
		matrixTranspose(W2, W2Trans, h1, C);

		// dW1_1 = dscores * W2.T
		matrixMultiply(dscores, W2Trans, dW1_1, N, C, h1);

		float *dfcRelu = new float[N * h1 * sizeof(float)];
		dfcRelu = (float *)memcpy((void *)dfcRelu, (void *)fc, N * h1 * sizeof(float));
		derivativeReLU(dfcRelu, N, h1);

		float *dfc = new float[N * h1 * sizeof(float)];
		matrixElementMultiply(dW1_1, dfcRelu, dfc, N, h1);

		float *XTrans = new float[d * N * sizeof(float)];
		matrixTranspose(X, XTrans, N, d);

		matrixMultiply(XTrans, dfc, dW1, d, N, h1);

		delete[] dW1_1;
		delete[] W2Trans;
		delete[] dfc;
		delete[] dfcRelu;
		delete[] XTrans;
	}

	void predict(int N, int d, int C, int h1, float *X, int *y, float *W1, float *W2, int* predictions) {
		int *dev_pred;
		cudaMalloc((void **)&dev_pred, N * sizeof(int));
		checkCUDAErrorWithLine("cudaMalloc dev_pred failed!");


	}


	void printArray2D(float *X, int nR, int nC) {
		for (int i = 0; i < nR; i++) {
			for (int j = 0; j < nC; j++)
				printf("%.4f ", X[i*nC + j]);
			printf("\n");
		}
		printf("\n");
	}

	// TODO: implement required elements for MLP sections 1 and 2 here

	void trainStep(int N, int d, int C, int h1, float alpha, float *X, int *y,  float *loss, float *W1, float *W2) {

		/*
			X -> N x d
			y -> N x 1
			W1 -> d x h1
			W2 -> h1 x C
		*/
		//printf("W1\n");
		//printArray2D(W1, d, h1);

		//printf("W2\n");
		//printArray2D(W2, h1, C);

		// first fully connected layer X2 = X*W1
		float *fc = new float[N * h1 * sizeof(float)];
		matrixMultiply(X, W1, fc, N, d, h1);
		//printf("FC:\n");
		//printArray2D(fc, N, h1);

		// Apply ReLU activation: X2 = ReLU(X2);
		float *X2 = new float[N * h1 * sizeof(float)];
		X2 = (float *) memcpy((void *)X2, (void *)fc, N * h1 * sizeof(float));
		reLU(X2, N, h1);
		//printf("ReLU:\n");
		//printArray2D(X2, N, h1);

		// calculate log_scores for softmax
		float *scores = new float[N * C * sizeof(float)];
		matrixMultiply(X2, W2, scores, N, h1, C);
		//printf("Log scores:\n");
		//printArray2D(scores, N, C);

		// calculate softmax probability: apply exp on all elements and normalize by sum of columns
		softmaxExp(scores, N, C);
		softmaxNormalize(scores, N, C);
		//printf("Softmax probabilities:\n");
		//printArray2D(scores, N, C);

		// calculate the loss
		calculateLoss(scores, N, C, y, loss);
		printf("Loss: %.4f\n", *loss);


		// **** BACKPROPAGATION STARTS *****
		// dL/dscores
		float *dscores = new float[N * C * sizeof(float)];
		softmaxDerivativeWrtScores(scores, y, dscores, N, C);

		//printf("dL/dscores\n");
		//printArray2D(dscores, N, C);

		float *dW2 = new float[h1 * C * sizeof(float)];
		calculateDerviativeW2(N, d, C, h1, dscores, X2, dW2);

		//printf("dW2\n");
		//printArray2D(dW2, h1, C);

		float *dW1 = new float[d * h1 * sizeof(float)];
		calculateDerviativeW1(N, d, C, h1, X, fc, W2, dscores, dW1);
		
		//printf("dW1\n");
		//printArray2D(dW1, d, h1);

		updateWeights(W1, dW1, W1, alpha, d, h1);
		//printf("W1\n");
		//printArray2D(W1, d, h1);

		updateWeights(W2, dW2, W2, alpha, h1, C);
		//printf("W2\n");
		//printArray2D(W2, h1, C);

		delete[] X2;
		delete[] scores;
		delete[] dscores;
		delete[] dW1;
		delete[] dW2;
		delete[] fc;

	}

	void predictAndAcc(int N, int d, int C, int h1, float *X, int *y,  float *W1, float *W2) {

	}


}
