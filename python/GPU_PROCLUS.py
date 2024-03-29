from torch.utils.cpp_extension import load
import numpy as np
import torch
import time
import os

from data.generator import *


def normalize(x):
    min_x = x.min(0, keepdim=True)[0]
    max_x = x.max(0, keepdim=True)[0]
    x_normed = (x - min_x) / (max_x - min_x)
    return x_normed


def load_vowel():
    return normalize(torch.from_numpy(np.loadtxt("data/vowel.dat", delimiter=',', skiprows=0)).float())


def load_glass():
    X = normalize(torch.from_numpy(np.loadtxt("data/glass.data", delimiter=',', skiprows=0)).float())
    X = X[:, 1:-1].clone()
    return X


def load_pendigits():
    X = normalize(torch.from_numpy(np.loadtxt("data/pendigits.tra", delimiter=',', skiprows=0)).float())
    X = X[:, :-1].clone()
    return X


def load_iris():
    X = normalize(torch.from_numpy(np.loadtxt("data/iris.data", delimiter=',', skiprows=0)).float())
    return X


def load_synt(d, n, cl, re, cl_d=None):
    if cl_d is None:
        cl_d = int(0.8 * d)

    name = "data_d" + str(d) + "_n" + str(n) + "_cl" + str(cl) + "_cl_d" + str(cl_d) + "_re" + str(re)

    if not os.path.exists('data/gen/'):
        os.makedirs('data/gen/')

    file = 'data/gen/' + name + '.dat'

    if not os.path.isfile(file):
        print("generating new data set")
        os.system("data/gen/cluster_generator -n" + str(n + 1) + " -d" + str(d) + " -k" + str(cl_d)
                  + " -m" + str(cl) + " -f" + str(0.99) + " " + file)

    print("loading data set...")
    d = []
    with open(file, 'r') as fp:
        line = fp.readline()
        for cnt, line in enumerate(fp):
            d.append([float(value) for value in line.split(' ')])
    print("data set loaded!")
    return normalize(torch.tensor(d))


def load_synt_gauss(d, n, cl, std, cl_n=None, cl_d=None, noise=0.01, re=0):
    if cl_d is None:
        cl_d = int(0.8 * d)

    if cl_n is None:
        cl_n = int((1. - noise) * n / cl)

    subspace_clusters = []
    for i in range(cl):
        subspace_clusters.append([cl_n, cl_d, 1, std])

    X, _ = generate_subspacedata_permuted(n, d, subspace_clusters)

    return normalize(torch.from_numpy(X)).float()


def load_skyserver_0_5x0_5():
    m_list = []
    for r in ["(0_0.5_0_0.5)"]:
        m_list.append(
            torch.from_numpy(np.loadtxt("data/real/skyserver/result " + r + ".csv", delimiter=',', skiprows=1)).float())
    m = torch.cat(m_list)
    return normalize(m)


def load_skyserver_1x1():
    m_list = []
    for r in ["(10_11_10_11)"]:
        m_list.append(
            torch.from_numpy(np.loadtxt("data/real/skyserver/result " + r + ".csv", delimiter=',', skiprows=2)).float())
    m = torch.cat(m_list)
    return normalize(m)


def load_skyserver_2x2():
    m_list = []
    for r in ["(10_11_10_11)", "(11_12_10_11)", "(10_11_11_12)", "(11_12_11_12)"]:
        m_list.append(
            torch.from_numpy(np.loadtxt("data/real/skyserver/result " + r + ".csv", delimiter=',', skiprows=2)).float())
    m = torch.cat(m_list)
    return normalize(m)


def load_skyserver_5x5():
    m_list = []
    for r in ["(10_15_10_11)", "(10_15_11_12)", "(10_15_12_13)", "(10_15_13_14)", "(10_15_14_15)"]:
        m_list.append(
            torch.from_numpy(np.loadtxt("data/real/skyserver/result " + r + ".csv", delimiter=',', skiprows=2)).float())
    m = torch.cat(m_list)
    return normalize(m)


def load_skyserver_10x10():
    m_list = []
    for r in ["(10_15_10_11)", "(10_15_11_12)", "(10_15_12_13)", "(10_15_13_14)", "(10_15_14_15)",
              "(10_15_15_16)", "(10_15_16_17)", "(10_15_17_18)", "(10_15_18_19)", "(10_15_19_20)",
              "(15_20_10_11)", "(15_20_11_12)", "(15_20_12_13)", "(15_20_13_14)", "(15_20_14_15)",
              "(15_20_15_16)", "(15_20_16_17)", "(15_20_17_18)", "(15_20_18_19)", "(15_20_19_20)"]:
        m_list.append(
            torch.from_numpy(np.loadtxt("data/real/skyserver/result " + r + ".csv", delimiter=',', skiprows=2)).float())
    m = torch.cat(m_list)
    return normalize(m)


t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
impl = load(name="GPU_PROCLUS17",
            sources=[
                "src/map/GPU_PROCLUS_map.cpp",
                "src/algorithms/PROCLUS.cpp",
                "src/algorithms/GPU_PROCLUS.cu",
                "src/utils/util.cpp",
                "src/utils/mem_util.cpp",
                "src/utils/gpu_util.cu"
            ], extra_cuda_cflags=["-w", "--std=c++14", "-arch=compute_75"], extra_cflags=["-w", "--std=c++17", "-fopenmp"], with_cuda=True)

print("Finished compilation, took: %.4fs" % (time.time() - t0))


def PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug)

def PROCLUS_parallel(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_parallel(X, k, l, a, b, min_deviation, termination_rounds, debug)

def FAST_star_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug)

def FAST_star_PROCLUS_parallel(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_KEEP_parallel(X, k, l, a, b, min_deviation, termination_rounds, debug)

def FAST_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug)

def FAST_PROCLUS_parallel(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_SAVE_parallel(X, k, l, a, b, min_deviation, termination_rounds, debug)

def FAST_PROCLUS_multi(X, ks, ls, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds, debug)

def FAST_PROCLUS_multi_parallel(X, ks, ls, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_PARAM_parallel(X, ks, ls, a, b, min_deviation, termination_rounds, debug)


def GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug)


def GPU_FAST_star_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.GPU_PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug)


def GPU_FAST_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.GPU_PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug)


def GPU_FAST_PROCLUS_multi(X, ks, ls, a, b, min_deviation, termination_rounds):
    return impl.GPU_PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds)

def GPU_FAST_PROCLUS_multi_2(X, ks, ls, a, b, min_deviation, termination_rounds):
    return impl.GPU_PROCLUS_PARAM_2(X, ks, ls, a, b, min_deviation, termination_rounds)

def GPU_FAST_PROCLUS_multi_3(X, ks, ls, a, b, min_deviation, termination_rounds):
    return impl.GPU_PROCLUS_PARAM_3(X, ks, ls, a, b, min_deviation, termination_rounds)
