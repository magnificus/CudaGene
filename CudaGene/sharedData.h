#pragma once

// must 
#define NUM_ORGANISMS 1000000

#define DNA_LENGTH 100
#define DNA_TYPE int

#define INPUT_DATA_TYPE int
#define OUTPUT_DATA_TYPE int

#define DATA_LENGTH 100

struct FittingData {
	INPUT_DATA_TYPE *inputData;
	OUTPUT_DATA_TYPE *groundTruth;
};

struct Organism {
	DNA_TYPE DNA[DNA_LENGTH];
};
