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

AVERAGING_FACTOR = 10
SCALING_FACTOR = 1.25
UPPER_BOUND = 12
BASE_ITER = 256
BASE_RES = 2048
MAX_RES = 23150

stats = ['avg', 'min', 'max']
EXPERIMENTS = {"res" : BASE_RES, "iter" : BASE_ITER}
CUDA_KERNELS = ["cudaNaive", "cudaDP"]
# KERNELS = ["naive"] + CUDA_KERNELS
KERNELS = CUDA_KERNELS
EXP_DIR = "experiments"
OUT_DIR = "output"
BASE_CMD = "./main -h {} -w {} -m {} -k {}"


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
            if exp == 'res':
                val = min(val, MAX_RES)
            if val not in experiments[exp][kernel]:
                experiments[exp][kernel][val] = {}
                experiments[exp][kernel][val]['trial'] = []
            if exp == "res":
                cmd = BASE_CMD.format(val, val, BASE_ITER, kernel).split()
            else:
                cmd = BASE_CMD.format(BASE_RES, BASE_RES, val, kernel).split()
            print "[Run] [exp='{}'] [kernel='{}'] [val='{}']".format(exp,
                    kernel, val)
            for trial in range(AVERAGING_FACTOR):
                seconds = subprocess.check_output(cmd)
                print "[Process] [exp='{}'] [kernel='{}'] [val='{}'] [seconds='{}']".format(exp,
                                kernel, val, seconds)
                experiments[exp][kernel][val]['trial'].append(seconds)

            # Aggregate stats.
            experiments[exp][kernel][val]['avg'] = sum(float(second)
                    for second in experiments[exp][kernel][val]['trial']) / AVERAGING_FACTOR
            experiments[exp][kernel][val]['min'] = min(float(second)
                    for second in experiments[exp][kernel][val]['trial'])
            experiments[exp][kernel][val]['max'] = max(float(second)
                    for second in experiments[exp][kernel][val]['trial'])
            for stat in stats:
                print "['{}'='{}']".format(stat,
                        experiments[exp][kernel][val][stat])

# REVIEW josephz: This can be abstracted by a function of keys and vals.
print "[Graphing Running Time]"
for exp in experiments:
    plt.figure()
    plt.title('Running Time')
    plt.xlabel(exp)
    plt.ylabel('Time(s)')
    for kernel in KERNELS:
        for stat in stats:
            plt.scatter(experiments[exp][kernel].keys(),
                    [experiments[exp][kernel][val][stat] for val in experiments[exp][kernel]],
                    label="{}_{}".format(kernel, stat), color=next(palette))
    leg = plt.legend(frameon=True, loc="upper left")
    plt.savefig(os.path.join(EXP_DIR,
        "running_time_graph_exp_{}.png".format(exp)))

print "[Graphing M. Pixels per second]"
for exp in experiments:
    plt.figure()
    plt.title('M. Pixels per second')
    plt.xlabel(exp)
    plt.ylabel('MPix/s')

    vals = experiments[exp][kernel].keys()
    resolutions = {}
    for stat in stats:
        if exp == "res":
            resolutions[stat] = [float(val) ** 2 / 1000000 /
                    experiments[exp][kernel][val][stat] for val in vals]
        else:
            resolutions[stat] = [float(BASE_RES) ** 2 / 1000000 /
                    experiments[exp][kernel][val][stat] for val in vals]

    for kernel in KERNELS:
        for stat in resolutions:
            plt.scatter(vals, resolutions[stat], label="{}_{}".format(kernel, stat), color=next(palette))
    leg = plt.legend(frameon=True, loc="upper left")
    plt.savefig(os.path.join(EXP_DIR,
        "mpps_graph_exp_{}.png".format(exp)))


