#!/usr/bin/python

"""
run.py
---



"""

import os
import sys
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

palette = itertools.cycle(sns.color_palette())

AVERAGING_FACTOR = 5
SCALING_FACTOR = 1.25
UPPER_BOUND = 11
BASE_ITER = 256
BASE_RES = 2048
MAX_RES = 20000

EXPERIMENTS = {"res" : BASE_RES, "iter" : BASE_ITER}
CUDA_KERNELS = ["cudaNaive", "cudaDP"]
KERNELS = ["naive"] + CUDA_KERNELS
# KERNELS = CUDA_KERNELS
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
            val = int(EXPERIMENTS[exp] * SCALING_FACTOR ** i)
            val = min(val, MAX_RES)
            if val not in experiments[exp][kernel]:
                experiments[exp][kernel][val] = []
            output = os.path.join(OUT_DIR,
                    "kernel_{}_exp_{}_val_{}_mandelbrot_set.png".format(kernel, exp, val))
            if exp == "res":
                cmd = BASE_CMD.format(val, val, BASE_ITER, kernel, output).split()
            else:
                cmd = BASE_CMD.format(BASE_RES, BASE_RES, val, kernel, output).split()
            print "[Run] [exp='{}'] [kernel='{}'] [val='{}']".format(exp,
                    kernel, val)
            for trial in range(AVERAGING_FACTOR):
                seconds = subprocess.check_output(cmd)
                print "[Process] [exp='{}'] [kernel='{}'] [val='{}'] [seconds='{}']".format(exp,
                                kernel, val, seconds)
                experiments[exp][kernel][val].append(seconds)
            experiments[exp][kernel][val] = sum(float(second) for second in experiments[exp][kernel][val]) / AVERAGING_FACTOR

# REVIEW josephz: This can be abstracted by a function of keys and vals.
print "[Graphing Running Time]"
for exp in experiments:
    plt.figure()
    plt.title('Running Time')
    plt.xlabel(exp)
    plt.ylabel('Time(s)')
    for kernel in KERNELS:
        plt.scatter(experiments[exp][kernel].keys(),
                [experiments[exp][kernel][val]
                    for val in experiments[exp][kernel]], label=kernel, color=next(palette))
    leg = plt.legend(frameon=True, loc="upper left")
    plt.savefig(os.path.join(EXP_DIR,
        "running_time_graph_exp_{}.png".format(exp)))
    plt.show()

print "[Graphing M. Pixels per second]"
for exp in experiments:
    plt.figure()
    plt.title('M. Pixels per second')
    plt.xlabel(exp)
    plt.ylabel('MPix/s')

    vals = experiments[exp][kernel].keys()
    if exp == "res":
        resolutions = [float(val) ** 2 / 1000000 / experiments[exp][kernel][val] for val in vals]
    else:
        resolutions = [BASE_RES ** 2 / 1000000 / experiments[exp][kernel][val] for val in vals]

    for kernel in KERNELS:
        plt.scatter(vals, resolutions, label=kernel, color=next(palette))
    leg = plt.legend(frameon=True, loc="upper left")
    plt.savefig(os.path.join(EXP_DIR,
        "mpps_graph_exp_{}.png".format(exp)))
    plt.show()


