#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
			bool exception = false;
			try {
				timer().startCpuTimer();
			}
			catch (const std::runtime_error& ex) {
				exception = true;
			}

            
			if (n <= 0) return;
			if (n == 1) {
				odata[0] = 0;
				return;
			}
			odata[0] = 0;
			odata[1] = idata[0];


			for (int i = 2; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}

			if(!exception)
				timer().endCpuTimer();
	
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            
			int k = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[k] = idata[i];
					k++;
				}
			}

	        timer().endCpuTimer();
            return k;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			int *checkNonZero = new int[n];
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) checkNonZero[i] = 1;
				else checkNonZero[i] = 0;
			}

			int *prefixSum = new int[n];
			scan(n, prefixSum, checkNonZero);

			for (int i = 0; i < n; i++) {
				if (checkNonZero[i] != 0) {
					odata[prefixSum[i]] = idata[i];
				}
			}

			int count = checkNonZero[n - 1] == 0 ? prefixSum[n - 1] : prefixSum[n - 1] + 1;
	        timer().endCpuTimer();
            return count;
        }
    }
}
