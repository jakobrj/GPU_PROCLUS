from python.GPU_PROCLUS import *
import os
import sys
import time
import matplotlib.pyplot as plt

dist_lim = 250
scale_lim = 1000
figure_size = (1.5*1.75, 1.5*1.4)
marker_size = 1
linewidth = 0.5
font_size = 10#8
plt.rcParams.update({'font.family': "Times"})
plt.rcParams.update({'font.serif': "Times"})
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'mathtext.default': "sf"})
plt.rcParams['text.usetex'] = True

label_PROCLUS = "PROCLUS"
label_FAST_star_PROCLUS = "FAST*-PROCLUS"
label_FAST_PROCLUS = "FAST-PROCLUS"
label_FAST_PROCLUS_multi = "FAST-PROCLUS(multi-param)"
label_PROCLUS_parallel = "PROCLUS(multi-core)"
label_FAST_star_PROCLUS_parallel = "FAST*-PROCLUS(multi-core)"
label_FAST_PROCLUS_parallel = "FAST-PROCLUS(multi-core)"
label_FAST_PROCLUS_multi_parallel = "FAST-PROCLUS(multi-param,multi-core)"
label_GPU_PROCLUS = "GPU-PROCLUS"
label_GPU_FAST_star_PROCLUS = "GPU-FAST*-PROCLUS"
label_GPU_FAST_PROCLUS = "GPU-FAST-PROCLUS"
label_GPU_FAST_PROCLUS_multi1 = "GPU-FAST-PROCLUS(multi-param1)"
label_GPU_FAST_PROCLUS_multi2 = "GPU-FAST-PROCLUS(multi-param2)"
label_GPU_FAST_PROCLUS_multi3 = "GPU-FAST-PROCLUS(multi-param3)"

style_map = {
    label_PROCLUS: {"color": "#A12C23", "marker": "x", "linestyle": "dashed"},
    label_FAST_star_PROCLUS: {"color": "#00554D", "marker": "*", "linestyle": "dashed"},
    label_FAST_PROCLUS: {"color": "#0281BB", "marker": "1", "linestyle": "dashed"},
    label_FAST_PROCLUS_multi: {"color": "#0281BB", "marker": "2", "linestyle": "dashed"},
    label_PROCLUS_parallel: {"color": "#A12C23", "marker": "x", "linestyle": "dotted"},
    label_FAST_star_PROCLUS_parallel: {"color": "#00554D", "marker": "*", "linestyle": "dotted"},
    label_FAST_PROCLUS_parallel: {"color": "#0281BB", "marker": "1", "linestyle": "dotted"},
    label_FAST_PROCLUS_multi_parallel: {"color": "#0281BB", "marker": "2", "linestyle": "dotted"},
    label_GPU_PROCLUS: {"color": "#74A23D", "marker": "x", "linestyle": "solid"},
    label_GPU_FAST_star_PROCLUS: {"color": "#8BB3FF", "marker": "*", "linestyle": "solid"},
    label_GPU_FAST_PROCLUS: {"color": "#F19000", "marker": "1", "linestyle": "solid"},
    label_GPU_FAST_PROCLUS_multi1: {"color": "#F19000", "marker": "2", "linestyle": "solid"},
    label_GPU_FAST_PROCLUS_multi2: {"color": "#F19000", "marker": "3", "linestyle": "solid"},
    label_GPU_FAST_PROCLUS_multi3: {"color": "#F19000", "marker": "4", "linestyle": "solid"},
}

algorithms_1 = [(PROCLUS, label_PROCLUS), (FAST_star_PROCLUS, label_FAST_star_PROCLUS),
                (FAST_PROCLUS, label_FAST_PROCLUS),
                (PROCLUS_parallel, label_PROCLUS_parallel),
                (FAST_star_PROCLUS_parallel, label_FAST_star_PROCLUS_parallel),
                (FAST_PROCLUS_parallel, label_FAST_PROCLUS_parallel),
                (GPU_PROCLUS, label_GPU_PROCLUS), (GPU_FAST_star_PROCLUS, label_GPU_FAST_star_PROCLUS),
                (GPU_FAST_PROCLUS, label_GPU_FAST_PROCLUS)]

algorithms_2 = [(PROCLUS, label_PROCLUS), (FAST_star_PROCLUS, label_FAST_star_PROCLUS),
                (FAST_PROCLUS, label_FAST_PROCLUS), (FAST_PROCLUS_multi, label_FAST_PROCLUS_multi),
                (PROCLUS_parallel, label_PROCLUS_parallel),
                (FAST_star_PROCLUS_parallel, label_FAST_star_PROCLUS_parallel),
                (FAST_PROCLUS_parallel, label_FAST_PROCLUS_parallel),
                (FAST_PROCLUS_multi_parallel, label_FAST_PROCLUS_multi_parallel),
                (GPU_PROCLUS, label_GPU_PROCLUS), (GPU_FAST_star_PROCLUS, label_GPU_FAST_star_PROCLUS),
                (GPU_FAST_PROCLUS, label_GPU_FAST_PROCLUS), (GPU_FAST_PROCLUS_multi, label_GPU_FAST_PROCLUS_multi1),
                (GPU_FAST_PROCLUS_multi_2, label_GPU_FAST_PROCLUS_multi2),
                (GPU_FAST_PROCLUS_multi_3, label_GPU_FAST_PROCLUS_multi3)]

algorithms_3 = [(GPU_PROCLUS, label_GPU_PROCLUS), (GPU_FAST_star_PROCLUS, label_GPU_FAST_star_PROCLUS),
                (GPU_FAST_PROCLUS, label_GPU_FAST_PROCLUS), (GPU_FAST_PROCLUS_multi, label_GPU_FAST_PROCLUS_multi1)]#,
                #(GPU_FAST_PROCLUS_multi_2, label_GPU_FAST_PROCLUS_multi2),
                #(GPU_FAST_PROCLUS_multi_3, label_GPU_FAST_PROCLUS_multi3),
                #(PROCLUS, label_PROCLUS), (FAST_star_PROCLUS, label_FAST_star_PROCLUS),
                #(FAST_PROCLUS, label_FAST_PROCLUS), (FAST_PROCLUS_multi, label_FAST_PROCLUS_multi),
                #(PROCLUS_parallel, label_PROCLUS_parallel),
                #(FAST_star_PROCLUS_parallel, label_FAST_star_PROCLUS_parallel),
                #(FAST_PROCLUS_parallel, label_FAST_PROCLUS_parallel),
                #(FAST_PROCLUS_multi_parallel, label_FAST_PROCLUS_multi_parallel)]


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


def run_space(algorithm, experiment, method, n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl,
              round,
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

        r = algorithm(X, k, l, a, b, min_deviation, termination_rounds)
        space = r[1] / 1024. / 1024.

        np.savez(run_file, space=space)

    else:
        data = np.load(run_file, allow_pickle=True)
        space = data["space"]

    return space


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
        if method == label_GPU_FAST_PROCLUS_multi1 or method == label_GPU_FAST_PROCLUS_multi2 or method == label_GPU_FAST_PROCLUS_multi3 or method == label_FAST_PROCLUS_multi or method == label_FAST_PROCLUS_multi_parallel:
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


def plot_multi(to_plt, xs, x_label, experiment, y_max=None, y_label='time in seconds', y_scale="log"):
    print(xs)

    plt.figure(figsize=figure_size)

    for algo_name, avg_running_times in to_plt:
        print(algo_name, avg_running_times)

    for algo_name, avg_running_times in to_plt:
        plt.plot(xs[:len(avg_running_times)], avg_running_times, color=style_map[algo_name]["color"],
                 marker=style_map[algo_name]["marker"],
                 linestyle=style_map[algo_name]["linestyle"], label=algo_name)

    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.yscale(y_scale)
    plt.grid(True, which="both", ls="-")
    if not y_max is None:
        plt.ylim(0.001, y_max)
    plt.tight_layout()
    plt.savefig("plots/" + experiment + ".pdf")
    plt.clf()


def plot_multi_legend(to_plt, xs, x_label, experiment, y_max=None, y_label='time in seconds', y_scale="log"):
    plt.figure(figsize=figure_size)

    for algo_name, avg_running_times in to_plt:
        plt.plot(xs[:len(avg_running_times)], avg_running_times, color=style_map[algo_name]["color"],
                 marker=style_map[algo_name]["marker"], linestyle=style_map[algo_name]["linestyle"], label=algo_name)
    plt.gcf().subplots_adjust(left=0.14)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))
    plt.yscale(y_scale)
    if not y_max is None:
        plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig("plots/" + experiment + "_legend.pdf")
    plt.clf()


def plot_speedup(to_plt, xs, x_label, experiment, y_max=None):
    print(to_plt)
    print(xs)

    _, base = to_plt[0]

    plt.figure(figsize=figure_size)

    for algo_name, avg_running_times in to_plt:
        plt.plot(xs[:len(avg_running_times)], np.array(base) / np.array(avg_running_times),
                 color=style_map[algo_name]["color"], marker=style_map[algo_name]["marker"],
                 linestyle=style_map[algo_name]["linestyle"], label=algo_name)

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
    _, base = to_plt[0]

    #plt.figure(figsize=figure_size)

    for algo_name, avg_running_times in to_plt:
        plt.plot([], [], color=style_map[algo_name]["color"], marker=style_map[algo_name]["marker"],
                 linestyle=style_map[algo_name]["linestyle"], label=algo_name)

    #plt.gcf().subplots_adjust(left=0.5)
    #plt.ylabel('factor of speedup')
    #plt.xlabel(x_label)
    plt.legend(loc='center', fontsize=font_size)#bbox_to_anchor=(1., 1.))
    if not y_max is None:
        plt.ylim(0, 8000)
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.savefig("plots/" + experiment + "_speedup_legend.pdf")
    plt.clf()


def run_rounds(experiment_name, algo, algo_name, f_run, n_=None, d_=None, k_=None, l_=None, a_=None, b_=None,
               min_deviation_=None, termination_rounds_=None, cl_=None, std_=None, dims_pr_cl_=None):
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()

    if n_ is not None:
        n = n_
    if d_ is not None:
        d = d_
    if k_ is not None:
        k = k_
    if l_ is not None:
        l = l_
    if a_ is not None:
        a = a_
    if b_ is not None:
        b = b_
    if min_deviation_ is not None:
        min_deviation = min_deviation_
    if termination_rounds_ is not None:
        termination_rounds = termination_rounds_
    if cl_ is not None:
        cl = cl_
    if std_ is not None:
        std = std_
    if dims_pr_cl_ is not None:
        dims_pr_cl = dims_pr_cl_

    avg = 0.
    for round in range(rounds):
        print("round:", round)
        avg += f_run(algo, experiment_name, algo_name, n, d, k, l, a, b, min_deviation,
                     termination_rounds, cl, std, dims_pr_cl, round)

    return avg / rounds


def run_experiment(experiment_name, algorithms, iterator, xs, x_label="number of points", y_label="time in seconds",
                   y_scale="log", y_max=scale_lim, speedup=True):
    n, d, k, l, a, b, min_deviation, termination_rounds, cl, std, dims_pr_cl, rounds = get_standard_params()

    print("running experiment:", experiment_name)

    if not os.path.exists('experiments_data/' + experiment_name + '/'):
        os.makedirs('experiments_data/' + experiment_name + '/')

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    to_plt = []
    for algo, algo_name in algorithms:
        print(algo_name)
        to_plt.append((algo_name, iterator(experiment_name, algo, algo_name)))

    plot_multi(to_plt, xs, x_label, experiment_name, y_max=y_max, y_label=y_label, y_scale=y_scale)
    plot_multi_legend(to_plt, xs, x_label, experiment_name, y_max=y_max, y_label=y_label, y_scale=y_scale)
    if speedup:
        plot_speedup(to_plt, xs, x_label, experiment_name, y_max=y_max)
        plot_speedup_legend(to_plt, xs, x_label, experiment_name, y_max=y_max)


def run_inc_n():
    experiment_name = "inc_n"

    algorithms = algorithms_1

    ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for n in reversed(ns):
            print("n:", n)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, n_=n))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms,  iterator, ns, x_label="number of points", y_label="time in seconds")


def run_inc_n_param():
    experiment_name = "inc_n_param"

    algorithms = algorithms_2

    ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for n in reversed(ns):
            print("n:", n)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run_param, n_=n))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, ns, x_label="number of points", y_label="time in seconds")

def run_inc_n_param_large():
    experiment_name = "inc_n_param_large"

    algorithms = algorithms_3

    ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for n in reversed(ns):
            print("n:", n)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run_param, n_=n))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, ns, x_label="number of points", y_label="time in seconds")


def run_inc_d():
    experiment_name = "inc_d"

    algorithms = algorithms_1

    ds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for d in reversed(ds):
            print("d:", d)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, d_=d))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, ds, x_label="number of dimensions", y_label="time in seconds")


def run_inc_d_param():
    experiment_name = "inc_d_param"

    algorithms = algorithms_2

    ds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for d in reversed(ds):
            print("d:", d)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run_param, d_=d))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, ds, x_label="number of dimensions", y_label="time in seconds")

def run_inc_d_param_large():
    experiment_name = "inc_d_param_large"

    algorithms = algorithms_3

    ds = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for d in reversed(ds):
            print("d:", d)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run_param, d_=d))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, ds, x_label="number of dimensions", y_label="time in seconds")


def run_inc_k():
    experiment_name = "inc_k"

    algorithms = algorithms_1

    ks = [5, 10, 15, 20, 25]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for k in reversed(ks):
            print("k:", k)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, k_=k))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator,ks, x_label="number of clusters", y_label="time in seconds")


def run_inc_l():
    experiment_name = "inc_l"

    algorithms = algorithms_1

    ls = [5, 10, 15]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for l in reversed(ls):
            print("l:", l)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, l_=l))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, ls, x_label="average number of dimensions",
                   y_label="time in seconds")


def run_inc_a():
    experiment_name = "inc_A"

    algorithms = algorithms_1

    As = [10, 20, 50, 100, 200, 300, 400, 500]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for A in reversed(As):
            print("A:", A)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, a_=A))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, As, x_label="constant A", y_label="time in seconds")


def run_inc_b():
    experiment_name = "inc_B"

    algorithms = algorithms_1

    Bs = [5, 10, 20, 50, 100]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for B in reversed(Bs):
            print("B:", B)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, b_=B))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, Bs, x_label="constant B", y_label="time in seconds")


def run_inc_dev():
    experiment_name = "inc_dev"

    algorithms = algorithms_1

    devs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for dev in reversed(devs):
            print("min_deviation:", dev)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, min_deviation_=dev))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, devs, x_label="$min_{deviation}$", y_label="time in seconds")


def run_inc_cl():
    experiment_name = "inc_cl"

    algorithms = [(PROCLUS, "PROCLUS"), (FAST_star_PROCLUS, "FAST*-PROCLUS"), (FAST_PROCLUS, "FAST-PROCLUS"),
                  (GPU_PROCLUS, "GPU-PROCLUS"), (GPU_FAST_star_PROCLUS, "GPU-FAST*-PROCLUS"),
                  (GPU_FAST_PROCLUS, "GPU-FAST-PROCLUS")]

    cls = [5, 10, 15, 20, 25]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for cl in reversed(cls):
            print("cl:", cl)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, cl_=cl))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, cls, x_label="number of actual clusters",
                   y_label="time in seconds")


def run_inc_std():
    experiment_name = "inc_std"

    algorithms = algorithms_1

    stds = [5, 10, 15, 20, 25]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for std in reversed(stds):
            print("std:", std)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run, std_=std))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, stds, x_label="standard deviation", y_label="time in seconds")


def run_space_n():
    experiment_name = "space_n"

    algorithms = [(GPU_PROCLUS, label_GPU_PROCLUS), (GPU_FAST_star_PROCLUS, label_GPU_FAST_star_PROCLUS),
                  (GPU_FAST_PROCLUS, label_GPU_FAST_PROCLUS)]

    ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]

    def iterator(experiment_name, algo, algo_name):
        avgs = []
        for n in reversed(ns):
            print("n:", n)
            avgs.append(run_rounds(experiment_name, algo, algo_name, run_space, n_=n))
        return list(reversed(avgs))

    run_experiment(experiment_name, algorithms, iterator, ns, x_label="number of points", y_label="memory in MB",
                   y_scale="linear", y_max=600, speedup=False)

experiment = sys.argv[1]
if experiment == "all":
    run_inc_n()
    run_inc_d()
    run_inc_k()
    run_inc_l()
    run_inc_a()
    run_inc_b()
    run_inc_dev()
    run_inc_cl()
    run_inc_std()
    run_space_n()
    run_inc_n_param()
    run_inc_d_param()
elif experiment == "inc_n":
    run_inc_n()
elif experiment == "inc_d":
    run_inc_d()
elif experiment == "inc_k":
    run_inc_k()
elif experiment == "inc_l":
    run_inc_l()
elif experiment == "inc_cl":
    run_inc_cl()
elif experiment == "inc_std":
    run_inc_std()
elif experiment == "large":
    run_inc_n_param_large()
    run_inc_d_param_large()
elif experiment == "large_n":
    run_inc_n_param_large()
elif experiment == "large_d":
    run_inc_d_param_large()
elif experiment == "param":
    run_inc_n_param()
    run_inc_d_param()
elif experiment == "space_n":
    run_space_n()

# real world
