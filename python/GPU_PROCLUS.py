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
    for r in ["(0_0.5_0_0.5)", "(0_0.5_0.5_1)", "(0.5_1_0_0.5)", "(0.5_1_0.5_1)"]:
        m_list.append(
            torch.from_numpy(np.loadtxt("data/real/skyserver/result " + r + ".csv", delimiter=',', skiprows=1)).float())
    m = torch.cat(m_list)
    return normalize(m)


t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
impl = load(name="GPU_PROCLUS11",
            sources=[
                "src/map/GPU_PROCLUS_map.cpp",
                "src/algorithms/PROCLUS.cpp",
                "src/algorithms/GPU_PROCLUS.cu",
                "src/utils/util.cpp",
                "src/utils/mem_util.cpp",
                "src/utils/gpu_util.cu"
            ], extra_cuda_cflags=["-w", "--allow-unsupported-compiler"], extra_cflags=["-w"], with_cuda=True)

print("Finished compilation, took: %.4fs" % (time.time() - t0))


def PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug)


def PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug)


def PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug)


def PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds, debug=False):
    return impl.PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds, debug)


def GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.GPU_PROCLUS(X, k, l, a, b, min_deviation, termination_rounds, debug)


def GPU_PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.GPU_PROCLUS_KEEP(X, k, l, a, b, min_deviation, termination_rounds, debug)


def GPU_PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug=False):
    return impl.GPU_PROCLUS_SAVE(X, k, l, a, b, min_deviation, termination_rounds, debug)


def GPU_PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds):
    return impl.GPU_PROCLUS_PARAM(X, ks, ls, a, b, min_deviation, termination_rounds)
