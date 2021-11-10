from python.GPU_PROCLUS import *

import os
import time
import numpy as np
import matplotlib.pyplot as plt


def run(method, X):
    print(X.size())
    k = 10
    l = 5
    a = min(100, X.shape[0] // k)
    b = 10
    min_deviation = 0.7
    termination_rounds = 5

    rounds = 10

    total = 0.
    for _ in range(rounds):
        print("round:", _)
        for k_i in [k + 1, k, k - 1]:
            for l_i in [l + 1, l, l - 1]:
                t0 = time.time()
                method(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                t1 = time.time()
                total += t1 - t0
    avg = total / (rounds * 9)
    return avg


def run_param(method, X):
    print(X.size())
    k = 10
    l = 5
    a = min(100, X.shape[0] // k)
    b = 10
    min_deviation = 0.7
    termination_rounds = 5

    rounds = 10

    total = 0.
    for _ in range(rounds):
        print("round:", _)
        t0 = time.time()
        method(X, [k + 1, k, k - 1], [l + 1, l, l - 1], a, b, min_deviation, termination_rounds)
        t1 = time.time()
        total += t1 - t0
    avg = total / (rounds * 9)
    return avg


if not os.path.exists('plots/'):
    os.makedirs('plots/')

X = load_vowel()

k = 5
l = 5
a = 40
b = 10
min_deviation = 0.7
termination_rounds = 5

# do one run just to get the GPU started and get more correct measurements
labels = ["glass", "vowel", "pendigits", "sky 1x1", "sky 5x5", "sky 10x10"]
ra = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 5))
width = 1. / 5.

PROCLUS_times = []
FAST_PROCLUS_multi_times = []
GPU_PROCLUS_times = []
GPU_FAST_PROCLUS_multi_times = []
run(GPU_PROCLUS, load_glass())

for load_data in [load_glass, load_vowel, load_pendigits, load_skyserver_1x1, load_skyserver_5x5, load_skyserver_10x10]:
    PROCLUS_times.append(run(PROCLUS, load_data()))
    FAST_PROCLUS_multi_times.append(run_param(FAST_PROCLUS_multi, load_data()))
    GPU_PROCLUS_times.append(run(GPU_PROCLUS, load_data()))
    GPU_FAST_PROCLUS_multi_times.append(run_param(GPU_FAST_PROCLUS_multi, load_data()))


rects1 = ax.bar(ra - 3 * width / 2, PROCLUS_times, width=width, label="PROCLUS")
rects3 = ax.bar(ra - width / 2, FAST_PROCLUS_multi_times, width=width, label="FAST-PROCLUS")
rects4 = ax.bar(ra + width / 2, GPU_PROCLUS_times, width=width, label="GPU-PROCLUS")
rects6 = ax.bar(ra + 3 * width / 2, GPU_FAST_PROCLUS_multi_times, width=width, label="GPU-FAST-PROCLUS")

ax.set_xticks(ra)
ax.set_xticklabels(labels)

plt.ylabel('time in seconds')

ax.legend()
plt.rc('font', size=11)
plt.yscale("log")
fig.tight_layout()
plt.savefig("plots/real.pdf")
plt.show()
