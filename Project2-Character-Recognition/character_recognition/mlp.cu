#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

#define blockSize 16

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	void printArray2D(float *dev_X, int nR, int nC) {

		float* X = new float[nR * nC];
		cudaMemcpy(X, dev_X, nR * nC * sizeof(float), cudaMemcpyDeviceToHost);

		for (int i = 0; i < nR; i++) {
			for (int j = 0; j < nC; j++)
				printf("%.4f ", X[i*nC + j]);
			printf("\n");
		}
		printf("\n");

		delete[] X;
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

	__global__ void kernMax(float *A, int *max, int N, int C) {
		int tx = (blockDim.x * blockIdx.x) + threadIdx.x;

		if (tx >= N) return;

		int maxIndex = 0;
		float maxVal = A[tx*C + 0];
		for (int i = 1; i < C; i++) {
			float val = A[tx*C + i];
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

	void softmaxExp(float *dev_A, int m, int n) {

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernExp << <dimGrid, dimBlock >> > (dev_A, m, n);
	}

	void softmaxNormalize(float *dev_A, int m, int n) {
		// TODO: Should be parallelized

		float *A = new float[m * n];
		cudaMemcpy(A, dev_A, m*n * sizeof(float), cudaMemcpyDeviceToHost);

		for (int i = 0; i < m; i++) {
			float sum = 0;
			for (int j = 0; j < n; j++)
				sum += A[i*n + j];

			for (int j = 0; j < n; j++)
				A[i*n + j] /= sum;
		}

		cudaMemcpy(dev_A, A, m*n * sizeof(float), cudaMemcpyHostToDevice);

		delete[] A;
	}

	void calculateLoss(float *dev_probs, int N, int C, int *dev_y, float* loss) {
		// TODO: Should be parallelized

		float *probs = new float[N * C];
		cudaMemcpy(probs, dev_probs, N * C * sizeof(float), cudaMemcpyDeviceToHost);

		int *y = new int[N];
		cudaMemcpy(y, dev_y, N * 1 * sizeof(int), cudaMemcpyDeviceToHost);

		float totalLoss = 0;
		for (int i = 0; i < N; i++) {
			int label = y[i];
			totalLoss += -log(probs[i*C + label]);
		}

		totalLoss /= N;

		*loss = totalLoss;

		delete[] probs;
		delete[] y;
	}

	void reLU(float *dev_A, int m, int n) {
		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernReLU<<<dimGrid, dimBlock>>>(dev_A, m, n);
	}

	void derivativeReLU(float *dev_A, int m, int n) {
		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernDerivativeReLU << <dimGrid, dimBlock >> > (dev_A, m, n);
	}

	void matrixMultiply(float *dev_A, float *dev_B, float *dev_C, int m, int n, int k) {

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (k + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_C, m, n, k);
	}

	void matrixElementMultiply(float *dev_A, float *dev_B, float *dev_C, int m, int n) {
		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernElementMatrixMultiply << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_C, m, n);
	}


	void matrixTranspose(float *dev_A, float *dev_B, int m, int n) {

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernMatrixTranspose << <dimGrid, dimBlock >> > (dev_A, dev_B, m, n);
	}

	void softmaxDerivativeWrtScores(float *dev_probs, int *dev_y, float *dev_dscores, int N, int C) {
		/*
		Calcluates dL/dscores . probs = softmax(scores)
		dL/dscores = probs[range(N), y] -= 1
		*/

		cudaMemcpy(dev_dscores, dev_probs, N * C * sizeof(float), cudaMemcpyDeviceToDevice);

		int gridRows = (N + blockSize - 1) / blockSize;
		int gridCols = (C + blockSize - 1) / blockSize;
		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);
		kernDerivativeLossScores << <dimGrid, dimBlock >> > (dev_probs, dev_y, dev_dscores, N, C);

	}

	void calculateDerviativeW2(int N, int d, int C, int h1, float *dscores, float *X2, float *dW2) {
		float *X2Trans;
		cudaMalloc((void**)&X2Trans, h1 * N * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc X2Trans failed");
		matrixTranspose(X2, X2Trans, N, h1);


		// dL/dW2 = X2.T * dscores (h1xN X NxC  = h1xC)
		matrixMultiply(X2Trans, dscores, dW2, h1, N, C);

		cudaFree(X2Trans);
		checkCUDAErrorWithLine("cudaFree X2Trans faild");
	}

	void calculateDerviativeW1(int N, int d, int C, int h1, float *X, float *fc, float *W2, float *dscores, float *dW1) {
		float *dW1_1;
		cudaMalloc((void**)&dW1_1, N * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc  failed");
		float *W2Trans;
		cudaMalloc((void**)&W2Trans, C * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc  failed");

		matrixTranspose(W2, W2Trans, h1, C);

		// dW1_1 = dscores * W2.T
		matrixMultiply(dscores, W2Trans, dW1_1, N, C, h1);

		float *dfcRelu;
		cudaMalloc((void**)&dfcRelu, N * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc  failed");

		cudaMemcpy(dfcRelu, fc, N * h1 * sizeof(float), cudaMemcpyDeviceToDevice);

		derivativeReLU(dfcRelu, N, h1);

		float *dfc;
		cudaMalloc((void**)&dfc, N * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc  failed");

		matrixElementMultiply(dW1_1, dfcRelu, dfc, N, h1);

		float *XTrans;
		cudaMalloc((void**)&XTrans, d * N * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc  failed");

		matrixTranspose(X, XTrans, N, d);

		matrixMultiply(XTrans, dfc, dW1, d, N, h1);

		cudaFree(dW1_1);
		checkCUDAErrorWithLine("cudaFree  failed");
		cudaFree(W2Trans);
		checkCUDAErrorWithLine("cudaFree  failed");
		cudaFree(dfcRelu);
		checkCUDAErrorWithLine("cudaFree  failed");
		cudaFree(dfc);
		checkCUDAErrorWithLine("cudaFree  failed");
		cudaFree(XTrans);
		checkCUDAErrorWithLine("cudaFree  failed");

	}

	void updateWeights(float *dev_A, float *dev_B, float *dev_C, float alpha, int m, int n) {

		int gridRows = (m + blockSize - 1) / blockSize;
		int gridCols = (n + blockSize - 1) / blockSize;

		dim3 dimGrid(gridCols, gridRows);
		dim3 dimBlock(blockSize, blockSize);

		kernElementMatrixAdd << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_C, alpha, m, n);
	}




	void allocateMemoryTrain(int N, int d, int C, int h1, float **fc, float **X2, float **scores, float **dscores, float **dW1, float **dW2) {

		cudaMalloc((void **)fc, N * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

		cudaMalloc((void **)X2, N * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

		cudaMalloc((void **)scores, N * C * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

		cudaMalloc((void **)dscores, N * C * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

		cudaMalloc((void **)dW2, h1 * C * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

		cudaMalloc((void **)dW1, d * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

	}

	void freeMemoryTrain(float *fc, float *X2, float *scores, float *dscores, float *dW1, float *dW2) {
		cudaFree(fc);
		checkCUDAErrorWithLine("cudaFree fc failed!");

		cudaFree(X2);
		checkCUDAErrorWithLine("cudaFree fc failed!");

		cudaFree(scores);
		checkCUDAErrorWithLine("cudaFree fc failed!");

		cudaFree(dscores);
		checkCUDAErrorWithLine("cudaaFree fc failed!");

		cudaFree(dW2);
		checkCUDAErrorWithLine("cudaFree fc failed!");

		cudaFree(dW1);
		checkCUDAErrorWithLine("cudaFree fc failed!");
	}

	void allocateMemoryInference(int N, int d, int C, int h1, float **fc, float **X2, float **scores) {

		cudaMalloc((void **)fc, N * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

		cudaMalloc((void **)X2, N * h1 * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

		cudaMalloc((void **)scores, N * C * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc fc failed!");

	}

	void freeMemoryInference(float *fc, float *X2, float *scores) {
		cudaFree(fc);
		checkCUDAErrorWithLine("cudaFree fc failed!");

		cudaFree(X2);
		checkCUDAErrorWithLine("cudaFree fc failed!");

		cudaFree(scores);
		checkCUDAErrorWithLine("cudaFree fc failed!");

	}

	// TODO: implement required elements for MLP sections 1 and 2 here

	void trainStep(int N, int d, int C, int h1, float alpha, float *X, int *y,  float *loss, float *W1, float *W2) {

		/*
			X -> N x d
			y -> N x 1
			W1 -> d x h1
			W2 -> h1 x C
		*/

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);


		// first fully connected layer X2 = X*W1
		float *fc, *X2, *scores, *dscores,  *dW1,  *dW2;
		allocateMemoryTrain(N, d, C, h1, &fc, &X2, &scores, &dscores, &dW1, &dW2);


		matrixMultiply(X, W1, fc, N, d, h1);
		//printf("Fc:\n");
		//printArray2D(fc, N, h1);

		// Apply ReLU activation: X2 = ReLU(X2);
		
		cudaMemcpy(X2, fc, N * h1 * sizeof(float), cudaMemcpyDeviceToDevice);
		reLU(X2, N, h1);
		//printf("ReLU:\n");
		//printArray2D(X2, N, h1);

		// calculate log_scores for softmax
		matrixMultiply(X2, W2, scores, N, h1, C);
		//printf("Scores:\n");
		//printArray2D(scores, N, C);

		// calculate softmax probability: apply exp on all elements and normalize by sum of columns
		softmaxExp(scores, N, C);
		softmaxNormalize(scores, N, C);
		//printf("Probs:\n");
		//printArray2D(scores, N, C);

		// calculate the loss
		calculateLoss(scores, N, C, y, loss);
		printf("Loss: %.4f\n", *loss);


		// **** BACKPROPAGATION STARTS *****
		softmaxDerivativeWrtScores(scores, y, dscores, N, C);

		calculateDerviativeW2(N, d, C, h1, dscores, X2, dW2);

		calculateDerviativeW1(N, d, C, h1, X, fc, W2, dscores, dW1);

		updateWeights(W1, dW1, W1, alpha, d, h1);

		updateWeights(W2, dW2, W2, alpha, h1, C);

		freeMemoryTrain(fc, X2, scores, dscores, dW1, dW2);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Time taken for one train step: %.2f\n", milliseconds);


	}


	void predict(int N, int d, int C, int h1, float *X, int *y, float *W1, float *W2, int* predictions) {

		float *fc, *X2, *scores;
		allocateMemoryInference(N, d, C, h1, &fc, &X2, &scores);

		matrixMultiply(X, W1, fc, N, d, h1);

		cudaMemcpy(X2, fc, N * h1 * sizeof(float), cudaMemcpyDeviceToDevice);
		reLU(X2, N, h1);

		matrixMultiply(X2, W2, scores, N, h1, C);

		// calculate softmax probability: apply exp on all elements and normalize by sum of columns
		softmaxExp(scores, N, C);
		softmaxNormalize(scores, N, C);


		int *dev_pred;
		cudaMalloc((void **)&dev_pred, N * C * sizeof(int));

		int numBlocks = (N + blockSize - 1) / blockSize;
		kernMax << <numBlocks, blockSize >> > (scores, dev_pred, N, C);
		cudaMemcpy(predictions, dev_pred, N * sizeof(int), cudaMemcpyDeviceToHost);

		freeMemoryInference(fc, X2, scores);
	}

	void predictAndAcc(int N, int d, int C, int h1, float *X, int *y,  float *W1, float *W2) {

		int* predictions = new int[N];
		predict(N, d, C, h1, X, y, W1, W2, predictions);

		for (int i = 0; i < N; i++) {
			printf("Predictions for %d example is %d\n", i + 1, predictions[i]+1);
		}

		int *host_y = new int[N];
		cudaMemcpy(host_y, y, N * 1 * sizeof(int), cudaMemcpyDeviceToHost);

		float accuracy = 0.0;
		for (int i = 0; i < N; i++) {
			accuracy += (predictions[i] == host_y[i]);
		}
		accuracy /= N;

		//printf("W1:\n");
		//printArray2D(W1, d, h1);

		//printf("W2:\n");
		//printArray2D(W2, h1, C);

		printf("\n\nAccuracy is %.4f\n\n", accuracy);
	}


}
