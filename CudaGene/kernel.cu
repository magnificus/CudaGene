#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "sharedData.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

extern "C" void
launch_addkernel(int size, int *c, const int *a, const int *b) {
    addKernel << <1, size >> > (c, a, b);
}


__global__ void fitnessCalculation(Organism *organisms, int lengthX, FittingData fittingData, float *fitnessResults) {
    //fitness

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    unsigned int id = y * lengthX + x;

    Organism organism = organisms[id];
    float totalFitness = 0;

    for (unsigned int i = 0; i < DATA_LENGTH; i++) {
        INPUT_DATA_TYPE currentData = fittingData.inputData[i]; // can be more later
        OUTPUT_DATA_TYPE truth = fittingData.groundTruth[i];
        totalFitness += abs(truth - organism.DNA[0]);
    }

    fitnessResults[id] =  totalFitness; 

}

extern "C" void
launch_fitnessCalculation(int numOrganisms, Organism *organisms, FittingData fittingData, float *fitnessResults) {
    int numOrganismsX = sqrt(numOrganisms);
    int numOrganismsY = numOrganisms / numOrganismsX;
    dim3 grid(numOrganismsX, numOrganismsY, 1);
    dim3 block(1, 1, 1);
    fitnessCalculation << < grid,block >> > (organisms, numOrganismsX, fittingData, fitnessResults);
}
