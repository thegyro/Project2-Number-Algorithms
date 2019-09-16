#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here

	void trainStep(int N, int d, int C, int h1, float alpha, float *X, int *y, float *loss, float *W1, float *W2);
	void matrixMultiply(float *A, float *B, float *C, int m, int n, int k);
}
