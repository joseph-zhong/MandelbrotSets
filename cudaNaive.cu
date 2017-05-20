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

int main(int argc, char *argv[]) {
  // Default operational values.
  int width;
  int height;
  int x;
  int y;
  int maxIterations;
  char *filename;
  FILE *fp;

  parseArgs(argc, argv, &width, &height, &maxIterations, &filename);

  if (VERBOSE) {
    printf("\n[main] OPERATING PARAMETERS\n");
    printf("-----------------------------\n");
    printf("\twidth: '%d'\n\theight: '%d'\n\tmaxIterations: '%d'\n\tfilename: '%s'\n\n",
        width, height, maxIterations, filename);
  }

  // Set filename for output image.
  fp = fopen(filename, "wb");

  // REVIEW josephz: These could be cmdline arguments but in order to
  // standardize the experiments, we will keep these constant for now.
  const float zoom = ZOOM_DEFAULT;
  const float xPos = X_POS_DEFAULT;
  const float yPos = Y_POS_DEFAULT;
  const float radius = RADIUS_DEFAULT;

  double pr, pi;                       //real and imaginary part of the pixel p
  double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old z

  // Write header to ppm file.
  fprintf(fp, "P6\n# Mandelbrot Set. \n%d %d\n255\n", width, height);



  // Begin timing.
  clock_t begin = clock(); 

  // Naively iterate through each pixel.
  for(y = 0; y < height; y++) {   // 3 Ops.
    for(x = 0; x < width; x++) {  // 3 Ops.
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
        endClock(begin);
        g_operations += 43;
        color(0, 0, 0, fp);
      }
      else {
        endClock(begin);
        g_operations += 56;

        double z = sqrt(newRe * newRe + newIm * newIm);
        int brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
        color(brightness, brightness, 255, fp);
      }
    }
  }
  
  reportClock();
  reportOperations();
  reportFlops();
  return EXIT_SUCCESS;
}







