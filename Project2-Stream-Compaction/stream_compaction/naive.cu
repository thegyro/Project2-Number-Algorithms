#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void prefixSum(int n, int d, int *odata, const int *idata) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;

			int min_k = 1 << (d-1);
			if (index < min_k || index >= n) return;

			odata[index] = idata[index - min_k] + idata[index];
		}

		__global__ void shiftRight(int n, int *odata, const int *idata) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;

			if (index >= n) return;

			if (index == 0) {
				odata[index] = 0;
				return;
			}

			odata[index] = idata[index-1];
		}

		void printxxx(int n, const int *a) {
			for (int i = 0; i < n; i++) {
				printf("%d ", a[i]);
			}
			printf("\n\n\n");
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

			int numThreads = 128;
			int numBlocks = (n + numThreads - 1) / numThreads;

			int d_max = ilog2ceil(n);

			int *dev_idata, *dev_odata1, *dev_odata2;
			cudaMalloc((void **)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

			cudaMalloc((void **)&dev_odata1, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata1 failed!");

			cudaMalloc((void **)&dev_odata2, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata2 failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_odata1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int *out1 = dev_idata;
			int *out2 = dev_odata1;


			for (int d = 1; d <= d_max; d++) {
				prefixSum<<<numBlocks, numThreads>>>(n, d, out2, out1);
				cudaMemcpy(out1, out2, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}

			shiftRight<<<numBlocks, numThreads>>>(n, out2, out1);

			cudaMemcpy(odata, out2, n * sizeof(int), cudaMemcpyDeviceToHost);
			

			cudaFree(dev_idata);
			cudaFree(dev_odata1);
			cudaFree(dev_odata2);

            timer().endGpuTimer();
        }
    }
}
