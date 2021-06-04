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

    rounds = 20

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
         (load_glass, load_vowel, load_pendigits, load_skyserver_1x1, load_skyserver_5x5, load_skyserver_10x10)]
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

labels = ["glass", "vowel", "pendigits", "sky 1x1", "sky 5x5", "sky 10x10"]
ra = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 5))
width = 1. / 7.

PROCLUS_times = runs(PROCLUS)
rects1 = ax.bar(ra - 5 * width / 2, PROCLUS_times, width=width, label="PROCLUS")
PROCLUS_KEEP_times = runs(PROCLUS_KEEP)
rects2 = ax.bar(ra - 3 * width / 2, PROCLUS_KEEP_times, width=width, label="PROCLUS-KEEP")
PROCLUS_SAVE_times = runs(PROCLUS_SAVE)
rects3 = ax.bar(ra - width / 2, PROCLUS_SAVE_times, width=width, label="PROCLUS-SAVE")

run(GPU_PROCLUS, load_glass())

GPU_PROCLUS_times = runs(GPU_PROCLUS)
rects4 = ax.bar(ra + width / 2, GPU_PROCLUS_times, width=width, label="GPU-PROCLUS")
GPU_PROCLUS_KEEP_times = runs(GPU_PROCLUS_KEEP)
rects5 = ax.bar(ra + 3 * width / 2, GPU_PROCLUS_KEEP_times, width=width, label="GPU-PROCLUS-KEEP")
GPU_PROCLUS_SAVE_times = runs(GPU_PROCLUS_SAVE)
rects6 = ax.bar(ra + 5 * width / 2, GPU_PROCLUS_SAVE_times, width=width, label="GPU-PROCLUS-SAVE")

print("PROCLUS:", PROCLUS_times)
print("PROCLUS-KEEP:", PROCLUS_KEEP_times)
print("PROCLUS-SAVE:", PROCLUS_SAVE_times)

print("GPU-PROCLUS:", GPU_PROCLUS_times)
print("GPU-PROCLUS-KEEP:", GPU_PROCLUS_KEEP_times)
print("GPU-PROCLUS-SAVE:", GPU_PROCLUS_SAVE_times)

np.savez("experiments_data/real.npz",
         PROCLUS_times=PROCLUS_times, PROCLUS_KEEP_times=PROCLUS_KEEP_times, PROCLUS_SAVE_times=PROCLUS_SAVE_times,
         GPU_PROCLUS_times=GPU_PROCLUS_times, GPU_PROCLUS_KEEP_times=GPU_PROCLUS_KEEP_times,
         GPU_PROCLUS_SAVE_times=GPU_PROCLUS_SAVE_times)

ax.set_xticks(ra)
ax.set_xticklabels(labels)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 1),
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)
# autolabel(rects6)
plt.ylabel('time in seconds')

ax.legend()
plt.rc('font', size=11)
plt.yscale("log")
fig.tight_layout()
plt.savefig("plots/real.pdf")
plt.show()
