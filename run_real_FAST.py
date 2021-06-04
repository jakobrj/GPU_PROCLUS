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
    avg = total / (rounds*9)
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

PROCLUS_times = []#runs(PROCLUS)
PROCLUS_PARAM_times = [] #runs_param(PROCLUS_PARAM)
GPU_PROCLUS_times = [] #runs(GPU_PROCLUS)
GPU_PROCLUS_PARAM_times = []#runs_param(GPU_PROCLUS_PARAM)
run(GPU_PROCLUS, load_glass())

for load_data in [load_glass, load_vowel, load_pendigits, load_skyserver_1x1, load_skyserver_5x5, load_skyserver_10x10]

    PROCLUS_times.append(run(PROCLUS, load_data()))
    PROCLUS_PARAM_times.append(run_param(PROCLUS_PARAM, load_data()))
    GPU_PROCLUS_times.append(run(GPU_PROCLUS, load_data()))
    GPU_PROCLUS_PARAM_times.append(run_param(GPU_PROCLUS_PARAM, load_data()))

    print("PROCLUS:", PROCLUS_times)
    print("PROCLUS-SAVE:", PROCLUS_PARAM_times)
    print("GPU-PROCLUS:", GPU_PROCLUS_times)
    print("GPU-PROCLUS-PARAM:", GPU_PROCLUS_PARAM_times)

    np.savez("experiments_data/real_param.npz",
             PROCLUS_times=PROCLUS_times, PROCLUS_PARAM_times=PROCLUS_PARAM_times,
             GPU_PROCLUS_times=GPU_PROCLUS_times,
             GPU_PROCLUS_PARAM_times=GPU_PROCLUS_PARAM_times)


rects1 = ax.bar(ra - 3 * width / 2, PROCLUS_times, width=width, label="PROCLUS")
rects3 = ax.bar(ra - width / 2, PROCLUS_PARAM_times, width=width, label="FAST-PROCLUS")
rects4 = ax.bar(ra + width / 2, GPU_PROCLUS_times, width=width, label="GPU-PROCLUS")
rects6 = ax.bar(ra + 3 * width / 2, GPU_PROCLUS_PARAM_times, width=width, label="GPU-FAST-PROCLUS")




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
