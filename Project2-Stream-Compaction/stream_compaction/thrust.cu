#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			int *dev_idata, *dev_odata;
			cudaMalloc((void **)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMalloc((void **)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			thrust::device_ptr<int> dev_idataItr(dev_idata);
			thrust::device_ptr<int> dev_odataItr(dev_odata);

			thrust::exclusive_scan(dev_idataItr, dev_idataItr + n, dev_odataItr);

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);


            timer().endGpuTimer();
        }
    }
}
