from python.GPU_PROCLUS import *

import os
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.size': 10})
figure_size = (1.5 * 3.36, 1.5 * 1.4)


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
labels = ["glass", "vowel", "pendigits", "sky 1x1", "sky 2x2", "sky 5x5"]
ra = np.arange(len(labels))
fig, ax = plt.subplots(figsize=figure_size)
width = 1. / 7.

run_file = "experiments_data/real.npz"

PROCLUS_times = [0.0778673701816135, 0.3760427316029867, 2.395080794228448, 8.587821976343792, 36.916119522518585,
                 261.5996938202116]
FAST_star_PROCLUS_times = []
FAST_PROCLUS_multi_times = [0.035419771406385636, 0.16441789468129475, 1.124128892686632, 4.478250458505419,
                            18.78843895594279, 139.96997718281216]
GPU_PROCLUS_times = [0.0024620109134250215, 0.002765899234347873, 0.0048781633377075195, 0.007174672020806207,
                     0.013782013787163629, 0.09205211533440484]
GPU_FAST_star_PROCLUS_times = []
GPU_FAST_PROCLUS_multi_times = [0.0010533783170912, 0.0009655343161688911, 0.002134320471021864,
                                0.0031624794006347655,
                                0.007363846566942003, 0.040379969278971355]
if os.path.exists(run_file):
    data = np.load(run_file, allow_pickle=True)
    PROCLUS_times = data["PROCLUS_times"]
    FAST_star_PROCLUS_times = data["FAST_star_PROCLUS_times"]
    FAST_PROCLUS_multi_times = data["FAST_PROCLUS_multi_times"]
    GPU_PROCLUS_times = data["GPU_PROCLUS_times"]
    GPU_FAST_star_PROCLUS_times = data["GPU_FAST_star_PROCLUS_times"]
    GPU_FAST_PROCLUS_multi_times = data["GPU_FAST_PROCLUS_multi_times"]

run(GPU_PROCLUS, load_glass())

for i, load_data in enumerate([load_glass, load_vowel, load_pendigits,
                               load_skyserver_1x1, load_skyserver_2x2, load_skyserver_5x5]):

    for times, algo in [(PROCLUS_times, PROCLUS), (FAST_star_PROCLUS_times, FAST_star_PROCLUS),
                        (FAST_PROCLUS_multi_times, FAST_PROCLUS_multi), (GPU_PROCLUS_times, GPU_PROCLUS),
                        (GPU_FAST_star_PROCLUS_times, GPU_FAST_star_PROCLUS),
                        (GPU_FAST_PROCLUS_multi_times, GPU_FAST_PROCLUS_multi)]:
        if i >= len(times):
            times.append(run(algo, load_data()))

            np.savez(run_file,
                     PROCLUS_times=PROCLUS_times,
                     FAST_star_PROCLUS_times=FAST_star_PROCLUS_times,
                     FAST_PROCLUS_multi_times=FAST_PROCLUS_multi_times,
                     GPU_PROCLUS_times=GPU_PROCLUS_times,
                     GPU_FAST_star_PROCLUS_times=GPU_FAST_star_PROCLUS_times,
                     GPU_FAST_PROCLUS_multi_times=GPU_FAST_PROCLUS_multi_times)

print("PROCLUS", PROCLUS_times)
print("FAST*-PROCLUS", FAST_star_PROCLUS_times)
print("FAST-PROCLUS", FAST_PROCLUS_multi_times)
print("GPU-PROCLUS", GPU_PROCLUS_times)
print("GPU-FAST*-PROCLUS", GPU_FAST_star_PROCLUS_times)
print("GPU-FAST-PROCLUS", GPU_FAST_PROCLUS_multi_times)

rects1 = ax.bar(ra - 5 * width / 2, PROCLUS_times, width=width, label="PROCLUS", c="#A12C23")
rects2 = ax.bar(ra - 3 * width / 2, FAST_star_PROCLUS_times, width=width, label="FAST*-PROCLUS", c="#00554D")
rects3 = ax.bar(ra - width / 2, FAST_PROCLUS_multi_times, width=width, label="FAST-PROCLUS", c="#0281BB")
rects4 = ax.bar(ra + width / 2, GPU_PROCLUS_times, width=width, label="GPU-PROCLUS", c="74A23D")
rects5 = ax.bar(ra + 3 * width / 2, GPU_FAST_star_PROCLUS_times, width=width, label="GPU-FAST*-PROCLUS", c="8BB3FF")
rects6 = ax.bar(ra + 5 * width / 2, GPU_FAST_PROCLUS_multi_times, width=width, label="GPU-FAST-PROCLUS", c="F19000")

ax.set_xticks(ra)
ax.set_xticklabels(labels)

plt.ylabel('time in seconds')

#ax.legend()
plt.ylim(0.001, 1000)
plt.yscale("log")
fig.tight_layout()
plt.savefig("plots/real.pdf")
plt.show()
