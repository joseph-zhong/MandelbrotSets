// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// common.cu
// ---
//
//  Common code.
//

#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

#include "defaults.h"
#include "common.h"

void color(int red, int green, int blue, FILE *fp) {
  unsigned char colors[3];
  colors[0] = (char) red;
  colors[1] = (char) green;
  colors[2] = (char) blue;
  fwrite(colors, 3, 1, fp);
}

void parseArgs(int argc, char *argv[], 
    int *width, int *height, int *maxIterations, 
    char **kernel, char **filename) {
  if (VERBOSE) {
    printf(USAGE, argv[0]);
    printf(EXAMPLE, argv[0]);
  }
  // Set default operation arguments if needed. Parse cmdline arguments
  // otherwise. 
  if (argc != 6) {
    // Set default operational values.
    *width = WIDTH_DEFAULT;
    *height = HEIGHT_DEFAULT;
    *maxIterations = MAX_ITER_DEFAULT;
    *kernel = NAIVE_HOST;
    *filename = NAIVE_OUT_DEFAULT;
  }
  else {
    *width = atoi(argv[1]);
    *height = atoi(argv[2]);
    *maxIterations = atoi(argv[3]);
    *kernel = argv[4];
    *filename = argv[5];
  }
  
  assert(*width > 0 && "Passed width must be positive.\n");
  assert(*height > 0 && "Passed height must be positive.\n");
  assert(*maxIterations > 0 && "Passed maxIterations must be positive.\n");
  assert(strcmp(*kernel, NAIVE_HOST) == 0 || strcmp(*kernel, CUDA_NAIVE) == 0 || strcmp(*kernel, CUDA_DP) == 0);
//  assert(access(*filename, W_OK) == 0 && "Passed output filename could not be opened for writing.\n");
}

void saveImage(const char *filename, int *values, int w, int h, int maxIterations) {
  png_bytep row;
  
  FILE *fp = fopen(filename, "wb");
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0); 
  png_infop info_ptr = png_create_info_struct(png_ptr);
  // exception handling
  setjmp(png_jmpbuf(png_ptr));
  png_init_io(png_ptr, fp);
  // write header (8 bit colour depth)
  png_set_IHDR(png_ptr, info_ptr, w, h,
               8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  // set title
  png_text title_text;
  title_text.compression = PNG_TEXT_COMPRESSION_NONE;
  title_text.key = "Title";
  title_text.text = "Mandelbrot set";
  png_set_text(png_ptr, info_ptr, &title_text, 1); 
  png_write_info(png_ptr, info_ptr);

  // write image data
  row = (png_bytep) malloc(3 * w * sizeof(png_byte));
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int r, g, b;
      mapValueToColor(&r, &g, &b, values[y * w + x], maxIterations);
      row[3 * x + 0] = (png_byte)r;
      row[3 * x + 1] = (png_byte)g;
      row[3 * x + 2] = (png_byte)b;
    }
    png_write_row(png_ptr, row);
  }
  png_write_end(png_ptr, NULL);

  fclose(fp);
  png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
  free(row);
}


void mapValueToColor(int *r, int *g, int *b, int value, int maxIterations) {
  int d = maxIterations / DIVIDE_FACTOR;
  if(value >= maxIterations) {
    *r = *g = *b = 0;
  } 
  else { 
    if(value < 0) {
      value = 0;
    }

    if(value <= d) {
      *r = *g = 0;
      *b = 128 + value * 127 / (d);
    } else {
      *b = 255;
      *r = *g = (value - d) * 255 / (maxIterations - d);
    } 
  }   
} 

__host__ __device__ int calculatePixelValue(int width, int height, int maxIterations,
    complexNum cMin, complexNum cMax, int x, int y, 
    const float radius) {
  // Plot bounds. 
  complexNum diff = cMax - cMin;

  // Determine pixel position.
  float xPos = (float) x / width * diff.a;
  float yPos = (float) y / height * diff.bi;

  // Initialize c and z.
  complexNum c = cMin + complexNum(xPos, yPos);
  complexNum z = c;

  int iterations = 0;
  while (iterations < maxIterations && absSquared(z) < radius) {
    z = z * z + c;
    iterations++;
  }
  return iterations;
}

__host__ __device__ int divup(int x, int y) { 
  return x / y + (x % y ? 1 : 0); 
}

