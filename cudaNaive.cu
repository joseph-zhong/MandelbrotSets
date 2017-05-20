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
//  Parallelism. See naive.c for an example of C implementation. You can also
//  reference http://rosettacode.org/wiki/Mandelbrot_set from Rosetta Stone.
//  
//
// Compile the program with:
//
//    make cudaNaive
//
//  Usage:
// 
//    ./cudaNaive <xmin> <xmax> <ymin> <ymax> <maxiter> <xres> <out.ppm>
//
//  Example:
//
//    ./cudaNaive 0.27085 0.27100 0.004640 0.004810 1000 1024 pic.ppm
//
//  The interior of Mandelbrot set is black, the levels are gray.
//  If you have very many levels, the picture is likely going to be quite
//  dark. You can postprocess it to fix the palette. For instance,
//  with ImageMagick you can do (assuming the picture was saved to pic.ppm):
//
//    convert -normalize pic.ppm pic.png
//
//  The resulting pic.png is still gray, but the levels will be nicer. You
//  can also add colors, for instance:
//
//    convert -negate -normalize -fill blue -tint 100 pic.ppm pic.png
//
//  See http://www.imagemagick.org/Usage/color_mods/ for what ImageMagick
//  can do. It can do a lot.
//


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#include "defaults.h"

int main(int argc, char* argv[]) {
  // The window in the plane.
  const double xmin = atof(argv[1]);
  const double xmax = atof(argv[2]);
  const double ymin = atof(argv[3]);
  const double ymax = atof(argv[4]);

  /* Maximum number of iterations, at most 65535. */
  const uint16_t maxiter = (unsigned short)atoi(argv[5]);

  /* Image size, width is given, height is computed. */
  const int xres = atoi(argv[6]);
  const int yres = (xres * (ymax-ymin)) / (xmax-xmin);

  /* The output file name */
  const char* filename = argv[7];

  /* Open the file and write the header. */
  FILE * fp = fopen(filename,"wb");
  char *comment="# Mandelbrot set";/* comment should start with # */


  // Parse cmdline arguments. Takes default values incorrect format is detected.
  if (argc != 8 || VERBOSE) {
    printf(USAGE, argv[0]);
    printf(EXAMPLE, argv[0]);

  }

  
  /*write ASCII header to the file*/
  fprintf(fp,
          "P6\n# Mandelbrot, xmin=%lf, xmax=%lf, ymin=%lf, ymax=%lf, maxiter=%d\n%d\n%d\n%d\n",
          xmin, xmax, ymin, ymax, maxiter, xres, yres, (maxiter < 256 ? 256 : maxiter));

  /* Precompute pixel width and height. */
  double dx=(xmax-xmin)/xres;
  double dy=(ymax-ymin)/yres;

  double x, y; /* Coordinates of the current point in the complex plane. */
  double u, v; /* Coordinates of the iterated point. */
  int i,j; /* Pixel counters */
  int k; /* Iteration counter */
  for (j = 0; j < yres; j++) {
    y = ymax - j * dy;
    for(i = 0; i < xres; i++) {
      double u = 0.0;
      double v= 0.0;
      double u2 = u * u;
      double v2 = v*v;
      x = xmin + i * dx;
      /* iterate the point */
      for (k = 1; k < maxiter && (u2 + v2 < 4.0); k++) {
            v = 2 * u * v + y;
            u = u2 - v2 + x;
            u2 = u * u;
            v2 = v * v;
      };
      /* compute  pixel color and write it to file */
      if (k >= maxiter) {
        /* interior */
        const unsigned char black[] = {0, 0, 0, 0, 0, 0};
        fwrite (black, 6, 1, fp);
      }
      else {
        /* exterior */
        unsigned char color[6];
        color[0] = k >> 8;
        color[1] = k & 255;
        color[2] = k >> 8;
        color[3] = k & 255;
        color[4] = k >> 8;
        color[5] = k & 255;
        fwrite(color, 6, 1, fp);
      };
    }
  }
  fclose(fp);
  return 0;
}



