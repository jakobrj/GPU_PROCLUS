from python.GPU_PROCLUS import *

import time
import numpy as np
import matplotlib.pyplot as plt


def run(method, X):
    k = 5
    l = 5
    a = 40
    b = 10
    min_deviation = 0.7
    termination_rounds = 5

    rounds = 10

    total = 0.
    for _ in range(rounds):
        t0 = time.time()
        method(X, k, l, a, b, min_deviation, termination_rounds)
        t1 = time.time()
        total += t1 - t0
    avg = total / rounds
    return avg


def runs(method):
    return [run(method, load_data()) for load_data in (load_glass, load_vowel, load_pendigits, load_skyserver_1x1)]


X = load_vowel()

k = 5
l = 5
a = 40
b = 10
min_deviation = 0.7
termination_rounds = 5

# do one run just to get the GPU started and get more correct measurements
GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds)

labels = ["glass", "vowel", "pendigits",  "sky 1x1"]
ra = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.20

rects1 = ax.bar(ra - 3 * width / 2, runs(PROCLUS), width=width, label="PROCLUS")
rects2 = ax.bar(ra - width / 2, runs(GPU_PROCLUS), width=width, label="GPU-PROCLUS")
rects3 = ax.bar(ra + width / 2, runs(GPU_PROCLUS_KEEP), width=width, label="GPU-PROCLUS-KEEP")
rects4 = ax.bar(ra + 3 * width / 2, runs(GPU_PROCLUS_SAVE), width=width, label="GPU-PROCLUS-SAVE")

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


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.ylabel('time in seconds')

ax.legend()
plt.rc('font', size=11)
plt.yscale("log")
fig.tight_layout()
plt.savefig("plots/real.pdf")
plt.show()
