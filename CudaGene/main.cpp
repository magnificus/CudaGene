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
        toReturn.groundTruth[i] = i * 2;
    }
    return toReturn;
}

Organism* setupInitialDNA() {
    Organism* organisms = (Organism*)malloc(ORGANISMS_SIZE);
    for (unsigned int i = 0; i < NUM_ORGANISMS; i++) {
        for (unsigned int j = 0; j < DNA_LENGTH; j++) {
            organisms[i].DNA[j] = 1;
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


cudaError_t evolve()
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
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
    while (true) {
        printf("generation %u\t", count++);
		cudaMemcpy(organismsOnGPU, organisms, ORGANISMS_SIZE, cudaMemcpyHostToDevice);

		launch_fitnessCalculation(NUM_ORGANISMS, organismsOnGPU, fittingDataOnGPU, fitnessResultsOnGPU);

		// Check for any errors launching the kernel
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


        //sort badabongus
        std::sort(sortedOrganisms.begin(), sortedOrganisms.end(), [&fitnessResults](const unsigned int a, const unsigned int b) {
           return fitnessResults[a] > fitnessResults[b];
            }
		);

        printf("Best: %f, Average: %f\n", fitnessResults[sortedOrganisms[0]], fitnessResults[sortedOrganisms[sortedOrganisms.size() / 2]]);

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
