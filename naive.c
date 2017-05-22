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
//  Each iteration the newZ is calculated from the square of the prevZ summed
//  with the current pixel value. We begin at the origin.
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
		const float zoom, const float yPos, const float xPos, const float radius, 
		FILE *fp) {
  // Begin clock. 
  // struct timespec tstart = {0,0};
  clock_t start = clock();

	double newRe, newIm, oldRe, oldIm, pr, pi;
	// Naively iterate through each pixel.
	for(int y = 0; y < height; y++) {   // 3 Ops.
	 for(int x = 0; x < width; x++) {  // 3 Ops.
		 // Calculate Z from the pixel location, zoom, and position values.
		 pr = 1.5 * (x - width / 2) / (0.5 * zoom * width) + xPos; // 8 Ops.
		 pi = (y - height / 2) / (0.5 * zoom * height) + yPos;     // 7 Ops.
		 newRe = newIm = oldRe = oldIm = 0; // 4 Ops. 
		 int i;
		 for(i = 0; i < maxIterations; i++) {  // 3 Ops.
			 oldRe = newRe;
			 oldIm = newIm;
			 newRe = oldRe * oldRe - oldIm * oldIm + pr; 
			 newIm = 2 * oldRe * oldIm + pi; // 11 Ops.
			 // Stop once our point exceeds the target radius.
			 if((newRe * newRe + newIm * newIm) > radius) break; // 4 Ops.
		 }   

		 // 43 Ops.  

		 // If iteration limit is reached, fill black. Colored otherwise.
		 if(i == maxIterations) {
			 color(0, 0, 0, fp);
		 }   
		 else {
			 double z = sqrt(newRe * newRe + newIm * newIm);
			 int brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
			 color(brightness, brightness, 255, fp);
		 }   
     g_operations += 56; 
	 }   
	}
  endClock(start);
}

