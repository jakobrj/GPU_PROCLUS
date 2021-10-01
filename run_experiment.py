from python.GPU_PROCLUS import *
import os
import sys
import time
import matplotlib.pyplot as plt

font_size = 20
dist_lim = 250
scale_lim = 1000
plt.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.serif': "Times New Roman"})

colors_4 = [
    "#74A23D",
    "#F19000",
    "#8BB3FF",
    "#E2D034",
]

markers_4 = [
    "x",
    "o",
    "*",
    "s",
]

linestyles_4 = [
    "solid",
    "solid",
    "solid",
    "solid",
]

# colors_8 = [
#     "#79211A",
#     "#00C6B4",
#     "#475BCC",
#     "#CBBB2F",
#     "#425D23",
#     "#FF57FF",
#     "#B56C00",
#     "#3AFFFF"
# ]

colors_8 = [
    "#00554D",
    "#A12C23",
    "#0281BB",
    "#CC33E7",

    "#74A23D",
    "#F19000",
    "#8BB3FF",
    "#E2D034",
]

markers_8 = [
    "x",
    "o",
    "*",
    "s",
    "x",
    "o",
    "*",
    "s",
]

linestyles_8 = [
    "dashed",
    "dashed",
    "dashed",
    "dashed",
    "solid",
    "solid",
    "solid",
    "solid",
]

colors_6 = [
    "#00554D",
    "#A12C23",
    "#0281BB",

    "#74A23D",
    "#F19000",
    "#8BB3FF",
]

markers_6 = [
    "x",
    "o",
    "*",
    "x",
    "o",
    "*",
]

linestyles_6 = [
    "dashed",
    "dashed",
    "dashed",
    "solid",
    "solid",
    "solid",
]


def get_standard_params():
    d = 15
    n = 64000
    cl = 10
    std = 5
    dims_pr_cl = 5

    k = cl
    l = dims_pr_cl
    a = 100
    b = 10
    min_deviation = 0.7
    termination_rounds = 5

    rounds = 20

    return n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds


def get_run_file(experiment, method, n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl,
                 generator, round):
    return "experiments_data/" + experiment + "/" + method + \
           "n" + str(n) + "d" + str(d) + "cl" + str(cl) + "std" + str(std) + "dims_pr_cl" + str(
        dims_pr_cl) + "generator_" + generator + \
           "k" + str(k) + "l" + str(l) + "A" + str(a) + "B" + str(b) + "min_deviation" + str(
        min_deviation) + "termination_rounds" + str(termination_rounds) + "_round" + str(round) + ".npz"


def run(algorithm, experiment, method, n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, round,
        generator="gaussian"):
    def load_synt_wrap(d, n, cl, std, cl_n=None, cl_d=None, noise=0.01, re=0):
        return load_synt(d, n, cl, re, cl_d=cl_d)

    gen = None
    if generator == "gaussian":
        gen = load_synt_gauss
    elif generator == "uniform":
        gen = load_synt_wrap

    run_file = get_run_file(experiment, method, n, d, k, l, a, b, min_deviation, termination_rounds, cl, std,
                            dims_pr_cl, generator, round)

    if not os.path.exists(run_file):
        X = gen(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        X_ = gen(n=10, d=2, cl=1, std=std, re=round, cl_d=1)

        algorithm(X_, 1, 1, 5, 4, min_deviation, termination_rounds)
        t0 = time.time()
        _ = algorithm(X, k, l, a, b, min_deviation, termination_rounds)
        t1 = time.time()
        running_time = t1 - t0

        np.savez(run_file, running_time=running_time)

    else:
        data = np.load(run_file, allow_pickle=True)
        running_time = data["running_time"]

    return running_time


def run_param(algorithm, experiment, method, n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl,
              round, generator="gaussian"):
    def load_synt_wrap(d, n, cl, std, cl_n=None, cl_d=None, noise=0.01, re=0):
        return load_synt(d, n, cl, re, cl_d=cl_d)

    gen = None
    if generator == "gaussian":
        gen = load_synt_gauss
    elif generator == "uniform":
        gen = load_synt_wrap

    run_file = get_run_file(experiment, method, n, d, k, l, a, b, min_deviation, termination_rounds, cl, std,
                            dims_pr_cl, generator, round)

    if not os.path.exists(run_file):
        X = gen(n=n, d=d, cl=cl, std=std, re=round, cl_d=dims_pr_cl)
        X_ = gen(n=10, d=2, cl=1, std=std, re=round, cl_d=1)

        GPU_PROCLUS(X_, 1, 1, 5, 4, min_deviation, termination_rounds)
        t0 = time.time()
        if method == "GPU-FAST-PROCLUS" or method == "GPU-FAST-PROCLUS_2" or method == "GPU-FAST-PROCLUS_3" or method == "FAST-PROCLUS":
            _ = algorithm(X, [k + 1, k, k - 1], [l + 1, l, l - 1], a, b, min_deviation, termination_rounds)
        else:
            for k_i in [k + 1, k, k - 1]:
                for l_i in [l + 1, l, l - 1]:
                    _ = algorithm(X, k_i, l_i, a, b, min_deviation, termination_rounds)
        t1 = time.time()
        running_time = t1 - t0

        np.savez(run_file, running_time=running_time)

    else:
        data = np.load(run_file, allow_pickle=True)
        running_time = data["running_time"]

    return running_time


def plot(avg_running_times, xs, x_label, experiment, y_max=None, y_label='time in seconds'):
    print(avg_running_times)
    print(xs)

    plt.rcParams.update({'font.size': font_size})
    plt.plot(xs[:len(avg_running_times)], avg_running_times, marker="x")
    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if not y_max is None:
        plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig("plots/" + experiment + ".pdf")
    plt.clf()


def plot_multi(to_plt, xs, x_label, experiment, y_max=None):
    print(to_plt)
    print(xs)

    colors = colors_8
    markers = markers_8
    linestyles = linestyles_8
    if len(to_plt) == 6:
        colors = colors_6
        markers = markers_6
        linestyles = linestyles_6
    if len(to_plt) == 4:
        colors = colors_4
        markers = markers_4
        linestyles = linestyles_4

    plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': font_size})
    i = 0
    for algo_name, avg_running_times in to_plt:
        plt.plot(xs[:len(avg_running_times)], avg_running_times, color=colors[i], marker=markers[i],
                 linestyle=linestyles[i], label=algo_name)
        i += 1
    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    if not y_max is None:
        plt.ylim(0.001, y_max)
    plt.tight_layout()
    plt.savefig("plots/" + experiment + ".pdf")
    plt.clf()


def plot_multi_legend(to_plt, xs, x_label, experiment, y_max=None):
    print(to_plt)
    print(xs)

    colors = colors_8
    markers = markers_8
    linestyles = linestyles_8
    if len(to_plt) == 6:
        colors = colors_6
        markers = markers_6
        linestyles = linestyles_6

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': font_size})
    i = 0
    for algo_name, avg_running_times in to_plt:
        plt.plot(xs[:len(avg_running_times)], avg_running_times, color=colors[i], marker=markers[i],
                 linestyle=linestyles[i], label=algo_name)
        i += 1
    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel('time in seconds')
    plt.xlabel(x_label)
    plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))
    plt.yscale("log")
    if not y_max is None:
        plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig("plots/" + experiment + "_legend.pdf")
    plt.clf()


def plot_speedup(to_plt, xs, x_label, experiment, y_max=None):
    print(to_plt)
    print(xs)

    colors = colors_8
    markers = markers_8
    linestyles = linestyles_8
    if len(to_plt) == 6:
        colors = colors_6
        markers = markers_6
        linestyles = linestyles_6

    _, base = to_plt[0]

    plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': font_size})
    i = 0
    for algo_name, avg_running_times in to_plt:
        plt.plot(xs[:len(avg_running_times)], np.array(base) / np.array(avg_running_times), color=colors[i],
                 marker=markers[i], linestyle=linestyles[i],
                 label=algo_name)
        i += 1
    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel('factor of speedup')
    plt.xlabel(x_label)
    plt.grid(True, which="both", ls="-")
    if not y_max is None:
        plt.ylim(0, 6000)
    plt.tight_layout()
    plt.savefig("plots/" + experiment + "_speedup.pdf")
    plt.clf()


def plot_speedup_legend(to_plt, xs, x_label, experiment, y_max=None):
    print(to_plt)
    print(xs)

    colors = colors_8
    markers = markers_8
    linestyles = linestyles_8
    if len(to_plt) == 6:
        colors = colors_6
        markers = markers_6
        linestyles = linestyles_6

    _, base = to_plt[0]

    plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': font_size})
    i = 0
    for algo_name, avg_running_times in to_plt:
        plt.plot([], [], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=algo_name)
        i += 1
    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel('factor of speedup')
    plt.xlabel(x_label)
    plt.legend(loc='upper right', bbox_to_anchor=(1., 1.))
    if not y_max is None:
        plt.ylim(0, 8000)
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.savefig("plots/" + experiment + "_speedup_legend.pdf")
    plt.clf()


def run_diff_n():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]  # , 2048000, 4096000, 8192000]

    print("running experiment: inc_n")

    if not os.path.exists('experiments_data/inc_n/'):
        os.makedirs('experiments_data/inc_n/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []
    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (PROCLUS_SAVE, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_PROCLUS_SAVE, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for n in reversed(ns):
            # cl = max(1, n//4000)
            print("n:", n, "cl:", cl)
            avg_running_time = 0.
            running_times = []
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_n", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time
                running_times.append(running_time)

            print(running_times)

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        avg_running_times = list(reversed(avg_running_times))
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ns, "number of points", "inc_n", y_max=scale_lim)
    plot_multi_legend(to_plt, ns, "number of points", "inc_n", y_max=scale_lim)
    plot_speedup(to_plt, ns, "number of points", "inc_n", y_max=scale_lim)
    plot_speedup_legend(to_plt, ns, "number of points", "inc_n", y_max=scale_lim)


def run_diff_d():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    print("running experiment: inc_d")

    if not os.path.exists('experiments_data/inc_d/'):
        os.makedirs('experiments_data/inc_d/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (PROCLUS_SAVE, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_PROCLUS_SAVE, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for d in ds:
            print("d:", d)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_d", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ds, "number of dimensions", "inc_d", y_max=scale_lim)
    plot_multi_legend(to_plt, ds, "number of dimensions", "inc_d", y_max=scale_lim)
    plot_speedup(to_plt, ds, "number of dimensions", "inc_d", y_max=scale_lim)
    plot_speedup_legend(to_plt, ds, "number of dimensions", "inc_d", y_max=scale_lim)


def run_diff_n_param():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]  # , 2048000, 4096000, 8192000]

    print("running experiment: inc_n_param")

    if not os.path.exists('experiments_data/inc_n_param/'):
        os.makedirs('experiments_data/inc_n_param/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []
    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (PROCLUS_SAVE, "FAST-PROCLUS"), (FAST_PROCLUS_multi, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_PROCLUS_SAVE, "GPU-FAST-PROCLUS"), (GPU_FAST_PROCLUS_multi, "GPU-FAST-PROCLUS"),
                            (GPU_FAST_PROCLUS_multi_2, "GPU-FAST-PROCLUS_2"), (GPU_FAST_PROCLUS_multi_3, "GPU-FAST-PROCLUS_3")]:
        avg_running_times = []
        for n in reversed(ns):
            # cl = max(1, n//4000)
            print("n:", n, "cl:", cl)
            avg_running_time = 0.
            running_times = []
            for round in range(rounds):
                print("round:", round)
                running_time = run_param(algo, "inc_n_param", algo_name, n, d, k, l, a, b, min_deviation,
                                         termination_rounds, cl, std, dims_pr_cl, round)
                avg_running_time += running_time
                running_times.append(running_time)

            print(running_times)

            avg_running_time /= (rounds*9)
            avg_running_times.append(avg_running_time)
        avg_running_times = list(reversed(avg_running_times))
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ns, "number of points", "inc_n_param", y_max=scale_lim)
    plot_multi_legend(to_plt, ns, "number of points", "inc_n_param", y_max=scale_lim)
    plot_speedup(to_plt, ns, "number of points", "inc_n_param", y_max=scale_lim)
    plot_speedup_legend(to_plt, ns, "number of points", "inc_n_param", y_max=scale_lim)


def run_diff_d_param():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ds = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    print("running experiment: inc_d_param")

    if not os.path.exists('experiments_data/inc_d_param/'):
        os.makedirs('experiments_data/inc_d_param/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (PROCLUS_SAVE, "FAST-PROCLUS"), (FAST_PROCLUS_multi, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_PROCLUS_SAVE, "GPU-FAST-PROCLUS"), (GPU_FAST_PROCLUS_multi, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for d in ds:
            print("d:", d)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run_param(algo, "inc_d_param", algo_name, n, d, k, l, a, b, min_deviation,
                                         termination_rounds, cl, std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= (rounds * 9)
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ds, "number of dimensions", "inc_d_param", y_max=scale_lim)
    plot_multi_legend(to_plt, ds, "number of dimensions", "inc_d_param", y_max=scale_lim)
    plot_speedup(to_plt, ds, "number of dimensions", "inc_d_param", y_max=scale_lim)
    plot_speedup_legend(to_plt, ds, "number of dimensions", "inc_d_param", y_max=scale_lim)


def run_diff_n_param_large():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000]

    print("running experiment: inc_n_param_large")

    if not os.path.exists('experiments_data/inc_n_param_large/'):
        os.makedirs('experiments_data/inc_n_param_large/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []
    for algo, algo_name in [(GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_PROCLUS_SAVE, "GPU-FAST-PROCLUS"), (GPU_FAST_PROCLUS_multi, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for n in reversed(ns):
            # cl = max(1, n//4000)
            print("n:", n, "cl:", cl)
            avg_running_time = 0.
            running_times = []
            for round in range(rounds):
                print("round:", round)
                running_time = run_param(algo, "inc_n_param_large", algo_name, n, d, k, l, a, b, min_deviation,
                                         termination_rounds, cl, std, dims_pr_cl, round)
                avg_running_time += running_time
                running_times.append(running_time)

            print(running_times)

            avg_running_time /= (rounds * 9)
            avg_running_times.append(avg_running_time)
        avg_running_times = list(reversed(avg_running_times))
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ns, "number of points", "inc_n_param_large", y_max=scale_lim)
    plot_multi_legend(to_plt, ns, "number of points", "inc_n_param_large", y_max=scale_lim)
    # plot_speedup(to_plt, ns, "number of points", "inc_n_param_large", y_max=scale_lim)


def run_diff_d_param_large():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ds = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

    print("running experiment: inc_d_param_large")

    if not os.path.exists('experiments_data/inc_d_param_large/'):
        os.makedirs('experiments_data/inc_d_param_large/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS"), (GPU_FAST_PROCLUS_multi, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for d in ds:
            print("d:", d)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run_param(algo, "inc_d_param_large", algo_name, n, d, k, l, a, b, min_deviation,
                                         termination_rounds, cl, std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= (rounds * 9)
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ds, "number of dimensions", "inc_d_param_large", y_max=scale_lim)
    plot_multi_legend(to_plt, ds, "number of dimensions", "inc_d_param_large", y_max=scale_lim)
    # plot_speedup(to_plt, ds, "number of dimensions", "inc_d_param_large", y_max=scale_lim)


def run_diff_k():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ks = [5, 10, 15, 20, 25]  # , 25, 30, 35, 40, 45, 50]

    print("running experiment: inc_k")

    if not os.path.exists('experiments_data/inc_k/'):
        os.makedirs('experiments_data/inc_k/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (FAST_PROCLUS, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for k in ks:
            print("k:", k)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_k", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ks, "number of clusters", "inc_k", y_max=scale_lim)
    plot_multi_legend(to_plt, ks, "number of clusters", "inc_k", y_max=scale_lim)


def run_diff_l():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    ls = [5, 10, 15]  # , 30, 35, 40, 45, 50]

    print("running experiment: inc_l")

    if not os.path.exists('experiments_data/inc_l/'):
        os.makedirs('experiments_data/inc_l/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (FAST_PROCLUS, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for l in ls:
            print("l:", l)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_l", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, ls, "average number of dimensions", "inc_l", y_max=scale_lim)
    plot_multi_legend(to_plt, ls, "average number of dimensions", "inc_l", y_max=scale_lim)


def run_diff_a():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    As = [10, 20, 50, 100, 200, 300, 400, 500]

    print("running experiment: inc_A")

    if not os.path.exists('experiments_data/inc_A/'):
        os.makedirs('experiments_data/inc_A/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (FAST_PROCLUS, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for a in As:
            print("A:", a)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_A", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, As, "constant A", "inc_A", y_max=scale_lim)
    plot_multi_legend(to_plt, As, "constant A", "inc_A", y_max=scale_lim)


def run_diff_b():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    Bs = [5, 10, 20, 50, 100]

    print("running experiment: inc_B")

    if not os.path.exists('experiments_data/inc_B/'):
        os.makedirs('experiments_data/inc_B/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (FAST_PROCLUS, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for b in Bs:
            print("B:", b)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_B", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, Bs, "constant B", "inc_B", y_max=scale_lim)
    plot_multi_legend(to_plt, Bs, "constant B", "inc_B", y_max=scale_lim)


def run_diff_dev():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    devs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

    print("running experiment: inc_dev")

    if not os.path.exists('experiments_data/inc_dev/'):
        os.makedirs('experiments_data/inc_dev/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (FAST_PROCLUS, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for min_deviation in devs:
            print("min_deviation:", min_deviation)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_dev", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, devs, "$min_{deviation}$", "inc_dev", y_max=scale_lim)
    plot_multi_legend(to_plt, devs, "$min_{deviation}$", "inc_dev", y_max=scale_lim)


def run_diff_cl():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    cls = [5, 10, 15, 20, 25]  # , 30, 35, 40, 45, 50]

    print("running experiment: inc_cl")

    if not os.path.exists('experiments_data/inc_cl/'):
        os.makedirs('experiments_data/inc_cl/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (FAST_PROCLUS, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for cl in cls:
            print("cl:", cl)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_cl", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, cls, "number of actual clusters", "inc_cl", y_max=scale_lim)
    plot_multi_legend(to_plt, cls, "number of actual clusters", "inc_cl", y_max=scale_lim)


def run_diff_std():
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()
    stds = [5, 10, 15, 20, 25]  # , 30, 35, 40, 45, 50]

    print("running experiment: inc_std")

    if not os.path.exists('experiments_data/inc_std/'):
        os.makedirs('experiments_data/inc_std/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []

    for algo, algo_name in [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"),
                            (FAST_PROCLUS, "FAST-PROCLUS"),
                            (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                            (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]:
        avg_running_times = []
        for std in stds:
            print("std:", std)
            avg_running_time = 0.
            for round in range(rounds):
                print("round:", round)
                running_time = run(algo, "inc_std", algo_name, n, d, k, l, a, b, min_deviation, termination_rounds, cl,
                                   std, dims_pr_cl, round)
                avg_running_time += running_time

            avg_running_time /= rounds
            avg_running_times.append(avg_running_time)
        to_plt.append((algo_name, avg_running_times))

    plot_multi(to_plt, stds, "standard deviation", "inc_std", y_max=scale_lim)
    plot_multi_legend(to_plt, stds, "standard deviation", "inc_std", y_max=scale_lim)


experiment = sys.argv[1]
if experiment == "all":
    run_diff_n_param_large()
    run_diff_d_param_large()
    run_diff_n_param()
    run_diff_d_param()
    run_diff_n()
    run_diff_d()
    run_diff_k()
    run_diff_l()
    run_diff_a()
    run_diff_b()
    run_diff_dev()
    run_diff_cl()
    run_diff_std()
elif experiment == "inc_n":
    run_diff_n()
elif experiment == "inc_d":
    run_diff_d()
elif experiment == "inc_k":
    run_diff_k()
elif experiment == "inc_l":
    run_diff_l()
elif experiment == "inc_cl":
    run_diff_cl()
elif experiment == "inc_std":
    run_diff_std()
elif experiment == "large":
    run_diff_n_param_large()
    run_diff_d_param_large()
elif experiment == "param":
    run_diff_n_param()
    run_diff_d_param()
# real world
