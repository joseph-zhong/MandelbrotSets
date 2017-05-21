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

__global__ void naiveMandelbrotSetsKernel(int height, int width, int maxIterations, 
     const float zoom, const float yPos, const float xPos, const float radius, 
     char *d_output, long int *d_operations) {
  double newRe, newIm, oldRe, oldIm, pr, pi; 
  
  // Naively iterate through each pixel.
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (x >= width || y >= height) return;
  
  int output_index = 3*width*y + x*3;
    
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
    *d_operations += 43;
    d_output[output_index] = (char) 0;
    d_output[output_index + 1] = (char) 0;
    d_output[output_index + 2] = (char) 0;
  }   
  else {
    *d_operations += 56;

    double z = sqrt(newRe * newRe + newIm * newIm);
    int brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
    d_output[output_index] = (char) brightness;
    d_output[output_index] = (char) brightness;
    d_output[output_index] = (char) 255;
  }
}






