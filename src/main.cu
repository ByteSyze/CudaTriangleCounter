/**
*
* Copyright (C) Tyler Hackett 2016
*
* CUDA Triangle Counter
*
* A quickly-written program to determine all possible combinations of
* valid triangles from a grid, allowing for certain coordinates of the
* grid to be marked as unusable.
*
* main.cu
*
* */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "main.h"

__global__ void countTriangles(uint2 *validPoints, int *count)
{
	//TODO
}

int main()
{
	int i = 0;
	int h_count = 0;

	int *h_countPtr = &h_count;
	int *d_count;

	//Calculate the size of the array of valid points.
	size_t size = (GRID_HEIGHT*GRID_WIDTH*sizeof(uint2) - sizeof(h_invalidPoints));

	uint2 *h_validPoints = (uint2 *)malloc(size);
	uint2 *d_validPoints;

	cudaMalloc((void **)&d_validPoints, size);
	cudaMalloc((void **)&d_count, sizeof(int));

	//Generate an array of all valid points within the bounds defined by GRID_WIDTH and GRID_HEIGHT.
	for (unsigned int x = 0; x < GRID_WIDTH; x++)
	{
		for (unsigned int y = 0; y < GRID_HEIGHT; y++)
		{
			uint2 p = { x, y };

			if (!isInvalidPoint(p))
			{
				h_validPoints[i] = p;
				i += 1;
			}
		}
	}

	printf("%d Valid Points\n", i);

    return 0;
}

/**
*
* Checks h_invalidPoints for a corresponding point, represented as a uint2.
*
* */
bool isInvalidPoint(uint2 p)
{
	for each(uint2 point in h_invalidPoints)
	{
		if (point.x == p.x && point.y == p.y)
			return true;
	}
	return false;
}

