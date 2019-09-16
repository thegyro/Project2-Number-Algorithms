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

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.01, 0.01);



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
			y[i - 1] = std::stoi(line) - 1 ;

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


	float *X = new float[52 * 101 * 101];
	int *y = new int[52];
	fillImage(X, y);


	int N = 52;
	int d = 101*101;
	int C = 52;
	int h1 = 1000;

	//float *X = new float[N * d * sizeof(float)];
	//int *y = new int[N * 1 * sizeof(int)];
	float *W1 = new float[d * h1 * sizeof(float)];
	float *W2 = new float[h1 * C * sizeof(float)];
	float loss_val = 0.0;
	float *loss = &loss_val;

	float alpha = 0.1;

	fillInputXOR(X, y);
	generateRandomWeights(W1, d, h1);
	generateRandomWeights(W2, h1, C);


	//printf("X:\n");
	//printArray2D(X, N, d);
	//printf("\n");
	//printf("W1:\n");
	//printArray2D(W1, d, h1);
	//printf("\n");
	//printf("W2:\n");
	//printArray2D(W2, h1, C);
	//printf("\n");

	for (int i = 1; i <= 100; i++) {
		printf("\n\nIteration %d\n\n", i);
		CharacterRecognition::trainStep(N, d, C, h1, alpha, X, y, loss, W1, W2);

	}

	CharacterRecognition::predictAndAcc(N, d, C, h1, X, y, W1, W2);

	//printArray2D(img, 101, 101);


	delete[] X;
	delete[] y;
	delete[] W1;
	delete[] W2;
}
