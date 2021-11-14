#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "sharedData.h"
#include <algorithm>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>


extern "C" void
launch_fitnessCalculation(int numOrganisms, Organism * organisms, FittingData fittingData, float* fitnessResults);


#define INPUT_DATA_SIZE (sizeof(INPUT_DATA_TYPE) * DATA_LENGTH)
#define GROUND_TRUTH_SIZE (sizeof(OUTPUT_DATA_TYPE) * DATA_LENGTH)

#define ORGANISMS_SIZE (sizeof(Organism) * NUM_ORGANISMS)
#define FITNESS_RESULT_SIZE (sizeof(float) * NUM_ORGANISMS)
#define SORTED_FITNESS_SIZE (sizeof(unsigned int) * NUM_ORGANISMS)

FittingData setupFittingData() {
    // this is where you define
    FittingData toReturn;
    toReturn.inputData = (INPUT_DATA_TYPE*) malloc(INPUT_DATA_SIZE);
    toReturn.groundTruth = (OUTPUT_DATA_TYPE*) malloc(GROUND_TRUTH_SIZE);

    for (unsigned int i = 0; i < DATA_LENGTH; i++) {
        toReturn.inputData[i] = i;
        toReturn.groundTruth[i] = 0;
    }
    return toReturn;
}

Organism* setupInitialDNA() {
    Organism* organisms = (Organism*)malloc(ORGANISMS_SIZE);
    for (unsigned int i = 0; i < NUM_ORGANISMS; i++) {
        for (unsigned int j = 0; j < DNA_LENGTH; j++) {
            organisms[i].DNA[j] = rand();
        }
    }
    return organisms;
}

void clearDNA(Organism *organisms) {
    free(organisms);
}

void clearFittingData(FittingData fittingData) {
    free(fittingData.groundTruth);
    free(fittingData.inputData);
}

float getRandomFloat() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}


cudaError_t evolve()
{
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    FittingData fittingData = setupFittingData();
    FittingData fittingDataOnGPU;
    Organism* organisms = setupInitialDNA();
    Organism* organismsOnGPU;
    float *fitnessResults = (float*) malloc(FITNESS_RESULT_SIZE);
    float *fitnessResultsOnGPU;
    std::vector<unsigned int> sortedOrganisms;
    for (unsigned int i = 0; i < NUM_ORGANISMS; i++) {
        sortedOrganisms.push_back(i);
    }

    // GPU allocation
    cudaMalloc((void**)&fittingDataOnGPU.inputData, INPUT_DATA_SIZE);
    cudaMalloc((void**)&fittingDataOnGPU.groundTruth, INPUT_DATA_SIZE);
    cudaMalloc((void**)&organismsOnGPU, ORGANISMS_SIZE);
    cudaMalloc((void**)&fitnessResultsOnGPU, FITNESS_RESULT_SIZE);

    // copy from local memory to GPU


    cudaMemcpy(fittingDataOnGPU.inputData, fittingData.inputData, INPUT_DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(fittingDataOnGPU.groundTruth, fittingData.groundTruth, INPUT_DATA_SIZE, cudaMemcpyHostToDevice);


    unsigned int count = 0;

    srand(static_cast <unsigned> (time(0)));

    // generational execution
    while (true) {
        printf("generation %u\t", count++);
		cudaMemcpy(organismsOnGPU, organisms, ORGANISMS_SIZE, cudaMemcpyHostToDevice);

		launch_fitnessCalculation(NUM_ORGANISMS, organismsOnGPU, fittingDataOnGPU, fitnessResultsOnGPU);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		
		cudaDeviceSynchronize();
        
        // retrieve the results
		cudaMemcpy(fitnessResults, fitnessResultsOnGPU, FITNESS_RESULT_SIZE, cudaMemcpyDeviceToHost);

        // GPU sort
        /*thrust::host_vector<unsigned int> hostSortedFitness(sortedOrganisms);
        thrust::device_vector<unsigned int> deviceSortedFitness = hostSortedFitness;
        thrust::sort(deviceSortedFitness.begin(), deviceSortedFitness.end());
        thrust::copy(deviceSortedFitness.begin(), deviceSortedFitness.end(), hostSortedFitness.begin());*/


        //sort slow mode on CPU
        std::sort(sortedOrganisms.begin(), sortedOrganisms.end(), [&fitnessResults](const unsigned int a, const unsigned int b) {
           return fitnessResults[a] < fitnessResults[b];
            }
		);

        printf("Best ID: %i, Value: %f, Average Value: %f\n", sortedOrganisms[0], fitnessResults[sortedOrganisms[0]], fitnessResults[sortedOrganisms[sortedOrganisms.size() / 2]]);
        printf("Best DNA: %i...\n", organisms[sortedOrganisms[0]].DNA[0]);

        std::vector<unsigned int> killedOrganisms;
        
        // determine who dies

        for (int i = 0; i < NUM_ORGANISMS; i++) {
            float modifier = (float(i) / float(NUM_ORGANISMS));
            float r = getRandomFloat();
            if (modifier * 2.0f * CULL_RATIO > r) {
                killedOrganisms.push_back(i);
            }
        }
  //      printf("Killed lads:\n");
		//for (unsigned int i = 0; i < killedOrganisms.size(); i++) {
		//	printf("%i,", killedOrganisms[i]);
		//}
  //      printf("\n");

        // repopulate

        for (unsigned int i : killedOrganisms) {
            unsigned int parent1 = rand() % (NUM_ORGANISMS);
            unsigned int parent2 = rand() % (NUM_ORGANISMS);

            // shuffle the DNA of the new organism
            for (unsigned int j = 0; j < DNA_LENGTH; j++) {
                float randN = getRandomFloat();
                if (randN < MUTATION_RATIO) {
                    organisms[sortedOrganisms[i]].DNA[j] = rand();
                }
                else if (randN < 0.5f) {
                    organisms[sortedOrganisms[i]].DNA[j] = organisms[parent1].DNA[j];
                }
                else {
                    organisms[sortedOrganisms[i]].DNA[j] = organisms[parent2].DNA[j];
                }

            }

        }


		//for (unsigned int i = 0; i < 3; i+=sortedOrganisms.size()/3) {
		//   printf("organism %i fitness: %f\n", sortedOrganisms[i], fitnessResults[sortedOrganisms[i]]);
		//   printf("organism %i fitness: %f\n", sortedOrganisms[i], fitnessResults[sortedOrganisms[i]]);
		//}

    }

    // cleanup
    clearFittingData(fittingData);
    clearDNA(organisms);
    free(fitnessResults);

    cudaFree(fittingDataOnGPU.inputData);
    cudaFree(fittingDataOnGPU.groundTruth);
    cudaFree(organismsOnGPU);
    cudaFree(fitnessResultsOnGPU);

    return cudaStatus;
}


int main()
{
    // Add vectors in parallel.
    cudaError_t cudaStatus = evolve();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
