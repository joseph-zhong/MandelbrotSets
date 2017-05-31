// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor: Tanner Schmidt
// Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets.
//
// naive.c
// --- 
//  
//  This is a naive C implementation.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "naive.h"
#include "common.h"
#include "defaults.h"
#include "metrics.h"

void naiveMandelbrotSets(int height, int width, int maxIterations, 
		const float radius, complexNum cMin, complexNum cMax, const char *filename) {
  
  int *h_output = (int*) malloc(sizeof(int) * height * width);

  // Begin clock. 
  clock_t start = clock();

	// Naively iterate through each pixel.
	for(int y = 0; y < height; ++y) {   
	 for(int x = 0; x < width; ++x) {  
     h_output[y * width + x] = calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius);
	 }   
	}
  endClock(start);
  saveImage(filename, h_output, width, height, maxIterations);
}

