CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 2 - Stream Compaction**

* Srinath Rajagopalan
  * [LinkedIn](https://www.linkedin.com/in/srinath-rajagopalan-07a43155), [twitter](https://twitter.com/srinath132)
* Tested on: Windows 10, i7-6700 @ 3.4GHz 16GB, Nvidia Quadro P1000 4GB (Moore 100B Lab)

### Scan and Stream Compaction

In this we project, I have implemented the exclusive scan and stream compaction algorithms for different configuratins 
1) CPU scan and CPU stream compaction
2) Naive GPU scan
3) Work-Efficient GPU scan and stream compaction
5) Scan from Thrust (to benchmark)

Scan and Compaction can be best understood with the following example:

* scan: 
  - goal: produce a prefix sum array of a given array (we only care about exclusive scan here)
  - input
    - [1 5 0 1 2 0 3]
  - output
    - [0 1 6 6 7 9 9]
* compact: 
  - goal: closely and neatly packed the elements != 0
  - input
    - [1 5 0 1 2 0 3]
  - output
    - [1 5 1 2 3]

### Performance Analysis
1) A block size that worked well for each of the configurations was `128`. I experimented with different size options from 256 to 512 and the performance remained similar. Decreasing the block size below `64` led to a drop in performance for all the GPU implementations. Thus, all the comparisons  are benchmarked by fixing block size as 128.

2) Performannce graphs comparing the differnet implementations are included below. For work-efficient compaction, I implemented two versions, one of which is inefficient but passes all the test cases for array sizes upto 2^25. This implementation calls the work-efficient scan function implemented as a part of the scan setup. But to stick to the API, it also does several `cudaMemcpy` from device to host and host to device which is not required if the interemediate scan buffers are never going to be needed on the CPU. However, the more efficient version is not passing the test case for not-power of two for array size `2^25`. 


	![](data/scan_perf_15.png)

	![](data/scan_perf_20.png)

	![](data/scan_perf_25.png)


	![](data/compact_perf_15.png)

	![](data/compact_perf_20.png)

	![](data/compact_perf_25.png)
