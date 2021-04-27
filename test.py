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

# X = load_synt_gauss(n=n, d=d, cl=cl, re=0, cl_d=dims_pr_cl, std=std)

n = X.shape[0]  # 100_000
d = X.shape[1]
cl = 3
std = 5
dims_pr_cl = 3

k = cl
l = dims_pr_cl
a = 8
b = 4
min_deviation = 0.7
termination_rounds = 5

rounds = 1

torch.cuda.synchronize()

experiment = sys.argv[1]

ls = [l + 1, l, l - 1]
ks = [k + 1, k, k - 1]

if experiment == "BASE":
    rs = GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("BASE", _)
        for k_i in ks:
            for l_i in ls:
                print("k:", k_i, "l", l_i)
                X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
                t0 = time.time()
                rs = GPU_PROCLUS(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "BASE_S":
    rs = GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "KEEP":
    rs = GPU_PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("KEEP", _)
        for k_i in ks:
            for l_i in ls:
                print("k:", k_i, "l", l_i)
                X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
                t0 = time.time()
                rs = GPU_PROCLUS_KEEP(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "KEEP_S":
    rs = GPU_PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "SAVE":
    rs = GPU_PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("SAVE", _)
        for k_i in ks:
            for l_i in ls:
                print("k:", k_i, "l", l_i)
                X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
                t0 = time.time()
                rs = GPU_PROCLUS_SAVE(X, k_i, l_i, a, b, min_deviation, termination_rounds)
                gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)

if experiment == "SAVE_S":
    rs = GPU_PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug=True)

if experiment == "CPU":
    gpu_avg_time = 0
    for _ in range(rounds):
        print("CPU", _)
        for k_i in ks:
            for l_i in ls:
                print("k:", k, "l", l)
                X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
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

if experiment == "PARAM":
    rs = GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds)
    gpu_avg_time = 0
    for _ in range(rounds):
        print("PARAM", _)
        X = load_synt_gauss(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        t0 = time.time()
        rs = GPU_PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds)
        gpu_avg_time += time.time() - t0
    gpu_avg_time /= rounds
    print("avg time: %.4fs" % gpu_avg_time)
