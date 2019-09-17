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
#include <string>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.01, 0.01);

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)



void printArray2D(float *X, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			printf("%.2f ", X[i*nC + j]);
		printf("\n");
	}
}

void printArray2D(int *X, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			printf("%d ", X[i*nC + j]);
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


void fillImage(float *X, int *y) {

	int j = 0;
	for (int i = 1; i <= 52; i++) {
		std::string fileName;
		if (i <= 9) {
			fileName = "C:\\\\Users\\sri07\\Desktop\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\0" + std::to_string(i) + "info.txt";
		}
		else {
			fileName = "C:\\\\Users\\sri07\\Desktop\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + std::to_string(i) + "info.txt";
		}

		std::ifstream myfile(fileName);
		std::string line;
		//std::cout << fileName << '\n';
		if (myfile.is_open())
		{
			std::getline(myfile, line);
			y[i-1] = std::stoi(line) - 1 ;

			std::getline(myfile, line);


			std::getline(myfile, line);
			std::string buf;
			std::stringstream ss(line);
			while (ss >> buf) {
				//std::cout << j << '\n';
				X[j++] = ((float)std::stoi(buf))/255.0;
			}


			myfile.close();
		}
	}

}

void fillImageRandom(float *X, int *y, int N, int d) {

	int j = 0;
	for (int i = 1; i <= N; i++) {
		//std::cout << fileName << '\n';
		
			y[i - 1] = i - 1;

			for(int k = 0; k < d; k++) {
				//std::cout << j << '\n';
				X[j++] = dis(gen);
			}

	}
}


void generateRandomWeights(float *W, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			W[i*nC + j] = dis(gen);
	}
}

int main(int argc, char* argv[]) {
    // Scan tests

	//int N = 4;
	//int d = 3;
	//int C = 2;
	//int h1 = 4;

	//float *X = new float[N * d * sizeof(float)];
	//int *y = new int[N * 1 * sizeof(int)];
	//float *W1 = new float[d * h1 * sizeof(float)];
	//float *W2 = new float[h1 * C * sizeof(float)];
	//float loss_val = 0.0;
	//float *loss = &loss_val;

	//float alpha = 0.5;

	//fillInputXOR(X, y);
	//generateRandomWeights(W1, d, h1);
	//generateRandomWeights(W2, h1, C);


	//printf("X:\n");
	//printArray2D(X, N, d);
	//printf("\n");
	//printf("W1:\n");
	//printArray2D(W1, d, h1);
	//printf("\n");
	//printf("W2:\n");
	//printArray2D(W2, h1, C);
	//printf("\n");

	//for (int i = 1; i <= 1000; i++) {
	//	printf("\n\nIteration %d\n\n", i);
	//	CharacterRecognition::trainStep(N, d, C, h1, alpha, X, y, loss, W1, W2);
	//	
	//}

	//CharacterRecognition::predictAndAcc(N, d, C, h1, X, y, W1, W2);

	int N = 52;
	int d = 101 * 101;
	int C = 52;
	int h1 = 10;

	float *X = new float[N * d];
	int *y = new int[N];
	//	
	////fillImageRandom(X, y, N, d);
	fillImage(X, y);
	//fillInputXOR(X, y);

	float *W1 = new float[d * h1 * sizeof(float)];
	float *W2 = new float[h1 * C * sizeof(float)];
	float loss_val = 0.0;
	float *loss = &loss_val;

	float alpha = 0.5;

	generateRandomWeights(W1, d, h1);
	generateRandomWeights(W2, h1, C);

	float *dev_X, *dev_W1, *dev_W2;
	int *dev_y;
	cudaMalloc((void **)&dev_X, N * d * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc failed!");
	cudaMalloc((void **)&dev_y, N * 1 * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc failed!");
	cudaMalloc((void **)&dev_W1, d * h1 * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc failed!");
	cudaMalloc((void **)&dev_W2, h1 * C * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc fc failed!");

	cudaMemcpy(dev_X, X, N * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, N * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_W1, W1, d * h1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_W2, W2, h1 * C * sizeof(float), cudaMemcpyHostToDevice);


	//std::ofstream myfile;
	//myfile.open("loss_curve_XOR.txt");

	for (int i = 1; i <= 100; i++) {
		printf("\n\nIteration %d\n\n", i);
		CharacterRecognition::trainStep(N, d, C, h1, alpha, dev_X, dev_y, loss, dev_W1, dev_W2);

		//myfile << i << " " << *loss << '\n';
	}

	CharacterRecognition::predictAndAcc(N, d, C, h1, dev_X, dev_y, dev_W1, dev_W2);

	//myfile.close();

	cudaFree(dev_X);
	checkCUDAErrorWithLine("cudaFree fc failed!");
	cudaFree(dev_y);
	checkCUDAErrorWithLine("cudaFree fc failed!");
	cudaFree(dev_W1);
	checkCUDAErrorWithLine("cudaFree fc failed!");
	cudaFree(dev_W2);
	checkCUDAErrorWithLine("cudaFree fc failed!");

	delete[] X;
	delete[] y;
	delete[] W1;
	delete[] W2;
}
