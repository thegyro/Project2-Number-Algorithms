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

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);


void printArray2D(float *X, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			printf("%.2f ", X[i*nC + j]);
		printf("\n");
	}
}

void fillInputXOR(float *X, int *y) {
	X[0] = 0.0, X[1] = 0.0, X[2] = 1.0;
	X[3] = 0.0, X[4] = 1.0, X[5] = 1.0;
	X[6] = 1.0, X[7] = 0.0, X[8] = 1.0;
	X[9] = 1.0, X[10] = 1.0; X[11] = 1.0;

	y[0] = 0;
	y[1] = 1;
	y[2] = 1;
	y[3] = 0;
}

void generateRandomWeights(float *W, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			W[i*nC + j] = dis(gen);
	}
}

int main(int argc, char* argv[]) {
    // Scan tests

	int N = 4;
	int d = 3;
	int C = 2;
	int h1 = 4;

	float *X = new float[N * d * sizeof(float)];
	int *y = new int[N * 1 * sizeof(int)];
	float *W1 = new float[d * h1 * sizeof(float)];
	float *W2 = new float[h1 * C * sizeof(float)];
	float loss_val = 0.0;
	float *loss = &loss_val;

	float alpha = 0.5;

	fillInputXOR(X, y);
	generateRandomWeights(W1, d, h1);
	generateRandomWeights(W2, h1, C);


	printf("X:\n");
	printArray2D(X, N, d);
	printf("\n");
	printf("W1:\n");
	printArray2D(W1, d, h1);
	printf("\n");
	printf("W2:\n");
	printArray2D(W2, h1, C);
	printf("\n");

	for (int i = 1; i <= 1000; i++) {
		printf("\n\nIteration %d\n\n", i);
		CharacterRecognition::trainStep(N, d, C, h1, alpha, X, y, loss, W1, W2);
		
	}

	delete[] X;
	delete[] y;
	delete[] W1;
	delete[] W2;
}
