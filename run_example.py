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

    rounds = 3

    total = 0.
    for _ in range(rounds):
        print("round:", _)
        t0 = time.time()
        method(X, k, l, a, b, min_deviation, termination_rounds)
        t1 = time.time()
        total += t1 - t0
    avg = total / rounds
    return avg


def runs(method):
    r = [run(method, load_data()) for load_data in
         (load_glass, load_vowel, load_pendigits, load_skyserver_1x1)]
    print(r)
    return r


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
GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds)

labels = ["glass", "vowel", "pendigits", "sky 1x1"]
ra = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 5))
width = 1. / 7.

PROCLUS_times = runs(PROCLUS)
rects1 = ax.bar(ra - 5 * width / 2, PROCLUS_times, width=width, label="PROCLUS")
FAST_star_PROCLUS_times = runs(PROCLUS_KEEP)
rects2 = ax.bar(ra - 3 * width / 2, FAST_star_PROCLUS_times, width=width, label="FAST*-PROCLUS")
FAST_PROCLUS_times = runs(PROCLUS_SAVE)
rects3 = ax.bar(ra - width / 2, FAST_PROCLUS_times, width=width, label="FAST-PROCLUS")

run(GPU_PROCLUS, load_glass())

GPU_PROCLUS_times = runs(GPU_PROCLUS)
rects4 = ax.bar(ra + width / 2, GPU_PROCLUS_times, width=width, label="GPU-PROCLUS")
GPU_FAST_star_PROCLUS_times = runs(GPU_PROCLUS_KEEP)
rects5 = ax.bar(ra + 3 * width / 2, GPU_FAST_star_PROCLUS_times, width=width, label="GPU-FAST*-PROCLUS")
GPU_FAST_PROCLUS_times = runs(GPU_PROCLUS_SAVE)
rects6 = ax.bar(ra + 5 * width / 2, GPU_FAST_PROCLUS_times, width=width, label="GPU-FAST-PROCLUS")

ax.set_xticks(ra)
ax.set_xticklabels(labels)

plt.ylabel('time in seconds')

ax.legend()
plt.rc('font', size=11)
plt.yscale("log")
fig.tight_layout()
plt.savefig("plots/example.png")
plt.show()
