// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor: Tanner Schmidt
// Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets.
//
// metrics.c
// ---
//  
//  This class will handle metrics measuring of the various experiments.
//

#include <stdio.h>

#include "metrics.h"

// Define extern variables.
double g_time_spent = 0.0;
long long int g_operations = 0;

void endClock(clock_t start) {
  g_time_spent += (long double) (clock() - start) / CLOCKS_PER_SEC;
}

void reportClock() {
  printf("Clock Report\n");
  printf("------------\n");
  printf("\tTotal Time(in seconds): %5.3f\n\n", g_time_spent / 1000);
}

void reportOperations() {
  printf("Operations Report\n");
  printf("-----------------\n");
  printf("\tTotal Operations: %lld\n\n", g_operations);
}

void reportFlops() {
  printf("Flops Report\n");
  printf("------------\n");
  printf("\tFlops: %5.3f\n\n", ((double) g_operations / max(1.0, g_time_spent)));
}

