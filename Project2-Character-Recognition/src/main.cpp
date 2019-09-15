/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"
#include <random>

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];


void printArray2D(float *X, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			printf("%.2f ", X[i*nC + j]);
		printf("\n");
	}
}

void fillInputXOR(float *X, int *y) {
	X[0] = 0.0, X[1] = 0.0;
	X[2] = 0.0, X[3] = 1.0;
	X[4] = 1.0, X[5] = 0.0;
	X[6] = 1.0, X[7] = 1.0;

	y[0] = 0;
	y[1] = 1;
	y[2] = 1;
	y[3] = 0;
}

void generateRandomWeights(float *W, int nR, int nC) {
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			W[i*nC + j] = dis(gen);
	}
}

int main(int argc, char* argv[]) {
    // Scan tests

	int N = 4;
	int d = 2;
	int C = 2;
	int h1 = 2;

	float *X = new float[N * d * sizeof(float)];
	int *y = new int[N * 1 * sizeof(int)];
	float *W1 = new float[d * h1 * sizeof(float)];
	float *W2 = new float[h1 * C * sizeof(float)];

	fillInputXOR(X, y);
	generateRandomWeights(W1, d, h1);
	generateRandomWeights(W2, h1, C);

	float *X2 = new float[N * h1 * sizeof(float)];
	CharacterRecognition::matrixMultiply(X, W1, X2, N, d, h1);

	printArray2D(X, N, d);
	printf("\n");
	printArray2D(W1, d, h1);
	printf("\n");
	printArray2D(W2, h1, C);
}
