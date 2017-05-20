###
# Joseph Zhong
# josephz@cs.washington.edu
# 19 May 2017
# CSE 599 Final Project
# Instructor: Tanner Schmidt
# Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets
# 
# makefile
# ---
#  This handles compilation for the project. 
#
###

FLAGS=-g -Wall -Wpedantic -O4 -lm


all: naive cudaNaive cudaDP

naive: naive.c metrics.o
	g++ $(FLAGS) metrics.o naive.c -o naive
	./naive
	xdg-open images/naive.ppm

cudaNaive: cudaNaive.cu metrics.o
	nvcc $(FLAGS) metrics.o cudaNaive.cu -o cudaNaive
	./cudaNaive
	xdg-open images/cudaNaive.ppm

cudaDP: cudaDP.cu metrics.o
	nvcc $(FLAGS) metrics.o cudaDP.cu -o cudaDP
	./cudaDP
	xdg-open images/cudaDP.ppm

metrics.o: metrics.c metrics.h
	g++ $(FLAGS) -c metrics.c 

clean:
	find \( -name '*.out' -or -name '*.o' -or -name '*~' -or -name '*.ppm' \) -delete
	rm naive cudaNaive cudaDP

