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

#include "defaults.h"
#include "common.h"
#include "cudaCommon.h"

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

void save_image(const char *filename, int *dwells, int w, int h, int maxIterations) {
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
      dwell_color(&r, &g, &b, dwells[y * w + x], maxIterations);
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


void dwell_color(int *r, int *g, int *b, int dwell, int maxIterations) {
  int DIVIDE_FACTOR = 4;
  int CUT_DWELL = maxIterations / DIVIDE_FACTOR;
  // black for the Mandelbrot set
  if(dwell >= maxIterations) {
    *r = *g = *b = 0;
  } 
  else { 
    // cut at zero
    if(dwell < 0) {
      dwell = 0;
    }

    *b = 255;
    // *r = *g = (dwell - CUT_DWELL) * 255 / (maxIterations - CUT_DWELL);

    if(dwell <= CUT_DWELL) {
      // from black to blue the first half
      *r = *g = 0;
      *b = 128 + dwell * 127 / (CUT_DWELL);
    } else {
      // from blue to white for the second half
      *b = 255;
      *r = *g = (dwell - CUT_DWELL) * 255 / (maxIterations - CUT_DWELL);
      // *r = *g = 256.0 * log2(1.75 + dwell - log2(log2(z))) / log2((double)maxIterations);
    } 
  }   
} 

