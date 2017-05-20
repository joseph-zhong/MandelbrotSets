// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaNaive.cu
// ---
// 
//  This is the naive CUDA C implementation without usage of Dynamic
//  Parallelism. 
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "common.h"
#include "defaults.h"
#include "metrics.h"

__global__ void naiveMandelbrotSets(int height, int width, int maxIterations, 
     const float zoom, const float yPos, const float xPos, const float radius, 
     FILE *fp) {
  // clock_t begin = clock();
  double newRe, newIm, oldRe, oldIm, pr, pi; 
  
  // Naively iterate through each pixel.
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int x = threadIdx.x + blockDim.x * blockIdx.x;

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
    // endClock(begin);
    // g_operations += 43; 
    // color(0, 0, 0, fp);
  }   
  else {
    // endClock(begin);
    // g_operations += 56;

    double z = sqrt(newRe * newRe + newIm * newIm);
    int brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
    // color(brightness, brightness, 255, fp);
  }
}






