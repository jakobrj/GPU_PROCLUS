from python.GPU_PROCLUS import *

import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# X = load_glass()
# X = load_vowel()
# X = load_pendigits()
X = load_iris()[:, :4]
# X = load_synt(20, 10000, 20, 0)


n = 10_000  # X.shape[0]  # 100_000
d = 15  # X.shape[1]
cl = 10
std = 5
dims_pr_cl = 5

k = cl
l = dims_pr_cl
a = 100
b = 50
min_deviation = 0.7
termination_rounds = 5

rounds = 5

X = load_synt_gauss(n=n, d=d, cl=cl, re=0, cl_d=dims_pr_cl, std=std)

torch.cuda.synchronize()

experiment = sys.argv[1]

ls = [l + 1, l, l - 1]
ks = [k + 1, k, k - 1]

if experiment == "GPU":
    rs = GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("GPU", _)
        X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        for k_i in ks:
            for l_i in ls:
                print("k:", k_i, "l", l_i)
                t0 = time.time()
                rs = GPU_PROCLUS(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "GPU_S":
    rs = GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "GPU_KEEP":
    rs = GPU_PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("GPU_KEEP", _)
        X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        for k_i in ks:
            for l_i in ls:
                print("k:", k_i, "l", l_i)
                t0 = time.time()
                rs = GPU_PROCLUS_KEEP(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "GPU_KEEP_S":
    rs = GPU_PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "GPU_SAVE":
    rs = GPU_PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("GPU_SAVE", _)
        X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        for k_i in ks:
            for l_i in ls:
                print("k:", k_i, "l", l_i)
                t0 = time.time()
                rs = GPU_PROCLUS_SAVE(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "GPU_SAVE_S":
    rs = GPU_PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "CPU":
    gpu_avg_time = 0
    for _ in range(rounds):
        print("CPU", _)
        X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        for k_i in ks:
            for l_i in ls:
                print("k:", k, "l:", l)
                t0 = time.time()
                rs = PROCLUS(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "CPU_S":
    rs = PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "CPU_KEEP_S":
    rs = PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "CPU_SAVE_S":
    rs = PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "CPU_PARAM":
    gpu_avg_time = 0
    for _ in range(rounds):
        print("CPU_PARAM", _)
        # X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        t0 = time.time()
        rs = PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds, debug=False)
        gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "CPU_PARAM_S":
    rs = PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "GPU_PARAM":
    rs = GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("GPU_PARAM", _)
        X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        t0 = time.time()
        rs = GPU_PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds)
        gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)
