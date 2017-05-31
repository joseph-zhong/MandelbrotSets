#!/usr/bin/python

"""
run.py
---



"""

import os
import sys
import subprocess
import matplotlib.pyplot as plt

SCALING_FACTOR = 2
UPPER_BOUND = 2
BASE_ITER = 256
BASE_RES = 4096
EXPERIMENTS = {"res" : BASE_RES, "iter" : BASE_ITER}
CUDA_KERNELS = ["cudaNaive", "cudaDP"]
KERNELS = ["naive"] + CUDA_KERNELS
EXP_DIR = "experiments"
OUT_DIR = "output"
BASE_CMD = "./main {} {} {} {} {}"


if not os.path.exists(EXP_DIR) or not os.path.isdir(EXP_DIR):
    os.mkdir(EXP_DIR)

experiments = {}
processes = []
for exp in EXPERIMENTS:
    if exp not in experiments:
        experiments[exp] = {}
    for kernel in KERNELS:
        if kernel not in experiments:
            experiments[exp][kernel] = {}
        for i in range(UPPER_BOUND):
            val = EXPERIMENTS[exp] * SCALING_FACTOR ** i
            if val not in experiments[exp][kernel]:
                experiments[exp][kernel][val] = 0
            output = os.path.join(OUT_DIR,
                    "kernel_{}_exp_{}_val_{}_mandelbrot_set.png".format(kernel, exp, val))
            if exp == "res":
                cmd = BASE_CMD.format(val, val, BASE_ITER, kernel, output).split()
            else:
                cmd = BASE_CMD.format(BASE_RES, BASE_RES, val, kernel, output).split()
            print "[Run] [exp='{}'] [kernel='{}'] [val='{}']".format(exp,
                    kernel, val)
            p = subprocess.Popen(cmd)
            processes.append((p, kernel, exp, val))


for p, kernel, exp, val in processes:
    seconds, err = p.communicate()
    print "[Gathering] [exp='{}'] [kernel='{}'] [val='{}'] [seconds='{}']".format(exp,
                    kernel, val, seconds)
    experiments[exp][kernel][val] = seconds


for exp in experiments:
    plt.figure()
    plt.subplot(111)

    for kernel in KERNELS:
        plt.scatter(experiments[exp][kernel], [experiments[exp][kernel][val] for
            val in experiments[exp][kernel]], label=kernel)
    plt.savefig(os.path.join(EXP_DIR, "kernel_{}_exp_{}_val_{}_graph.png"))
    plt.imshow()

