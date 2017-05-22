// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor: Tanner Schmidt
// Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets.
//
// naive.h
// --- 
//  
//  This is a naive C implementation.
//
//  Each iteration the newZ is calculated from the square of the prevZ summed
//  with the current pixel value. We begin at the origin.
//

#ifndef NAIVE_H
#define NAIVE_H

// Naive Mandelbrot Set implementation in C. 
void naiveMandelbrotSets(int height, int width, int maxIterations, 
		const float zoom, const float yPos, const float xPos, const float radius, 
		const char *filename);

#endif // NAIVE_H
