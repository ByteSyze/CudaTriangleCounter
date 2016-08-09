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
* main.h
*
* */

#define GRID_WIDTH 4
#define GRID_HEIGHT 4

__global__ void countTriangles(uint2 *validPoints, int *count);

const uint2 h_invalidPoints[] = { { 0, 0 } };

bool isInvalidPoint(uint2 p);
