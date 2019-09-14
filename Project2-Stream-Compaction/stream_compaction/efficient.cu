#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void resetZeros(int n, int *a) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;
			if (index >= n) return;
			a[index] = 0;
		}


		__global__ void upSweep(int n, int d, int *idata) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;

			int twoPowd1 = 1 << (d + 1);
			int twoPowd = 1 << d;


			if ((index % twoPowd1 != twoPowd1-1) || index >= n) return;

			int k = index - twoPowd1 + 1;
			idata[index] += idata[k + twoPowd - 1];
		}

		__global__ void downSweep(int n, int d, int *idata) {
			int index = (blockDim.x*blockIdx.x) + threadIdx.x;

			int twoPowd1 = 1 << (d + 1);
			int twoPowd = 1 << d;


			if ((index % twoPowd1 != twoPowd1 - 1) || index >= n) return;

			int k = index - twoPowd1 + 1;
			int t = idata[k + twoPowd - 1];
			idata[k + twoPowd - 1] = idata[index];
			idata[index] += t;
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
			bool exception = false;
			try {
				timer().startGpuTimer();
			} catch (const std::runtime_error& ex) {
				exception = true;
			}

			int *dev_idata;

			int numThreads = 128;
			int numBlocks = (n + numThreads - 1) / numThreads;

			int d_max = ilog2ceil(n);

			int twoPowN = 1 << d_max;
			if (n != twoPowN) {

				int diff = twoPowN - n;

				cudaMalloc((void **)&dev_idata, (n + diff) * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_odata1 failed!");

				resetZeros << <numBlocks, numThreads >> > (n + diff, dev_idata);

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				n = n + diff;
			} else {
				cudaMalloc((void **)&dev_idata, n * sizeof(int));
				checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

				cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			}

			for (int d = 0; d < d_max; d++) {
				upSweep<<<numBlocks, numThreads>>>(n, d, dev_idata);
			}

			// reset last element to zero
			int* zero = new int[1];
			zero[0] = 0;
			cudaMemcpy(dev_idata + n - 1, zero, sizeof(int), cudaMemcpyHostToDevice);

			
			for(int d = d_max-1; d >= 0; d--) {
				downSweep << <numBlocks, numThreads >> > (n, d, dev_idata);
			}


			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(dev_idata);

			if(!exception)
				timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

			int numThreads = 128;
			int numBlocks = (n + numThreads - 1) / numThreads;

			int *dev_checkZeros, *dev_sumIndices, *dev_odata, *dev_idata;

			cudaMalloc((void **) &dev_checkZeros, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_checkZeros failed!");
			cudaMalloc((void **) &dev_sumIndices, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_sumIndices failed!");
			cudaMalloc((void **)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMalloc((void **)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			StreamCompaction::Common::kernMapToBoolean<<<numBlocks, numThreads>>>(n, dev_checkZeros, dev_idata);
			
			int *checkZeros = new int[n];
			cudaMemcpy(checkZeros, dev_checkZeros, n * sizeof(int), cudaMemcpyDeviceToHost);

			//printxxx(n, checkZeros);

			int *sumIndices = new int[n];
			scan(n, sumIndices, checkZeros);

			cudaMemcpy(dev_sumIndices, sumIndices , n * sizeof(int), cudaMemcpyHostToDevice);

			StreamCompaction::Common::kernScatter<<<numBlocks, numThreads>>>(n, dev_odata, dev_idata, dev_checkZeros, dev_sumIndices);

			

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			

			int count = checkZeros[n - 1] == 0 ? sumIndices[n - 1] : sumIndices[n - 1] + 1;

			//delete[] checkZeros;
			//delete[] sumIndices;

			//printf("hey\n");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_checkZeros);
			cudaFree(dev_sumIndices);

            timer().endGpuTimer();
            return count;
        }
    }
}
