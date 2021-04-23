#include "PROCLUS.h"
#include "../utils/util.h"
#include <numeric>
#include <cassert>
#include <time.h>
#include <cstdio>
#include <fstream>
#include <cmath>

#define DEBUG false

float manhattan_segmental_distance(bool *D_i, at::Tensor data, int m_i, int m_j, int d) {
    float sum = 0.;
    int size = 0;
    float *x_m_i = data[m_i].data_ptr<float>();
    float *x_m_j = data[m_j].data_ptr<float>();
    for (int k = 0; k < d; k++) {
        if (D_i[k]) {
            sum += std::abs(x_m_i[k] - x_m_j[k]);
            size++;
        }
    }
    return sum / size;
}

void greedy(int *M, float *dist, float *new_dist, at::Tensor data, int *S, int Bk, int Ak, int d) {

//    int *M = new int[Bk];
    int rnd_start = Ak / 2;//std::rand() % Ak
    M[0] = S[rnd_start];
    compute_l2_norm_to_medoid(dist, data, S, M[0], Ak, d);

    for (int i = 1; i < Bk; i++) {
//        dist[M[i - 1]] = -1.;

        printf("dist:\n");
        print_array(dist, Ak);

        M[i] = S[argmax_1d(dist, Ak)];
        int m_i = M[i];

        printf("m_i: %d\n", m_i);

        compute_l2_norm_to_medoid(new_dist, data, S, m_i, Ak, d);
        index_wise_minimum(dist, new_dist, Ak);
    }
}

bool **find_dimensions(at::Tensor data, int **L, int *L_sizes, int *M, int k, int n, int d, int l) {
    //print_debug("find_dimensions - start\n", DEBUG);
    float **X = array_2d<float>(k, d);
    float **Z = array_2d<float>(k, d);
    float *Y = new float[k];
    bool **D = zeros_2d<bool>(k, d);
    float *sigma = new float[k];

    //print_debug("find_dimensions - init \n", DEBUG);

    //compute X,Y,Z

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            X[i][j] = 0;
        }
    }

    for (int i = 0; i < k; i++) {
        if (L_sizes[i] == 0) {

        } else {
            float *x_m_i = data[M[i]].data_ptr<float>();
            for (int p = 0; p < L_sizes[i]; p++) {
                int point = L[i][p];
                float *x_p = data[point].data_ptr<float>();
                for (int j = 0; j < d; j++) {
                    X[i][j] += std::abs(x_p[j] - x_m_i[j]);
                }
            }
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            if (L_sizes[i] == 0) {

            } else {
                X[i][j] /= L_sizes[i];
            }
        }
    }

    for (int i = 0; i < k; i++) {

        Y[i] = mean_1d(X[i], d);

        sigma[i] = 0;
        for (int j = 0; j < d; j++) {
            float sub = X[i][j] - Y[i];
            sigma[i] += sub * sub;
        }
        sigma[i] /= (d - 1);
        sigma[i] = std::sqrt(sigma[i]);

        for (int j = 0; j < d; j++) {
            if (sigma[i] == 0) //todo not good... case not defined
                Z[i][j] = 0;
            else
                Z[i][j] = (X[i][j] - Y[i]) / sigma[i];
        }
    }

    //# ensuring that we find atleast 2 for each and than the k*l #todo fast - sort first instead
    for (int i = 0; i < k; i++) {
        /*printf("Z[%d]: \n", i);
        print_array(Z[i], d);*/
        for (int _ = 0; _ < 2; _++) {
            int j = argmin_1d<float>(Z[i], d);
            /*printf("i,j: %d,%d\n", i, j);
            assert(("0<=j<d", j < d && j >= 0));*/
            Z[i][j] = std::numeric_limits<float>::max();
            D[i][j] = true;
        }
    }

    for (int _ = k * 2; _ < k * l; _++) {
        std::pair<int, int> *p_i_j = argmin_2d(Z, k, d);
        int i = p_i_j->first;
        int j = p_i_j->second;
        Z[i][j] = std::numeric_limits<float>::max();
        D[i][j] = true;
    }

    //print_debug("find_dimensions - end\n", DEBUG);

    for (int i = 0; i < k; i++) {
        delete X[i];
        delete Z[i];
    }
    delete X;
    delete Y;
    delete Z;
    delete sigma;

    return D;
}

int *assign_points(at::Tensor data, bool **D, int *M, int n, int d, int k) {
    int *C = new int[n];
    float **dist = array_2d<float>(n, k);
    for (int i = 0; i < k; i++) {

        float *tmp_dist = compute_l1_norm_to_medoid(data, M[i], D[i], n, d);
        for (int j = 0; j < n; j++) {
            dist[j][i] = tmp_dist[j];
        }
        delete tmp_dist;
    }

    for (int p = 0; p < n; p++) {
        int i = argmin_1d<float>(dist[p], k);
        C[p] = i;
    }

    for (int i = 0; i < k; i++) {
        C[M[i]] = i;
    }

    for (int j = 0; j < n; j++) {
        delete dist[j];
    }
    delete dist;

    return C;
}


float evaluate_cluster(at::Tensor data, bool **D, int *C, int n, int d, int k) {
    float **Y = zeros_2d<float>(k, d);
    float *w = new float[k];

    float **means = zeros_2d<float>(k, d);
    int *counts = zeros_1d<int>(k);

    for (int i = 0; i < n; i++) {
        counts[C[i]] += 1;
        float *x_i = data[i].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            if (D[C[i]][j]) {
                means[C[i]][j] += x_i[j];
            }
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            means[i][j] /= counts[i];
        }
    }

    for (int i = 0; i < n; i++) {
        float *x_i = data[i].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            if (D[C[i]][j]) {
                Y[C[i]][j] += std::abs(x_i[j] - means[C[i]][j]);
            }
        }
    }
    for (int j = 0; j < d; j++) {
        for (int i = 0; i < k; i++) {
            if (D[i][j]) {
                Y[i][j] /= counts[i];
            }
        }
    }

    for (int i = 0; i < k; i++) {
        w[i] = 0.;
        int size = 0;
        for (int j = 0; j < d; j++) {
            if (D[i][j]) {
                w[i] += Y[i][j];
                size++;
            }
        }
        w[i] /= size;
    }


    float sum = 0;
    for (int i = 0; i < k; i++) {
        sum += counts[i] * w[i];
    }

    for (int i = 0; i < k; i++) {
        delete Y[i];
        delete means[i];
    }
    delete Y;
    delete w;
    delete means;
    delete counts;

    return sum / n;
}

bool *bad_medoids(int *C, int k, float min_deviation, int n) {
    bool *bad = zeros_1d<bool>(k);
    int *sizes = zeros_1d<int>(k);

    for (int i = 0; i < n; i++) {
        sizes[C[i]] += 1;
    }

    int first = argmin_1d<int>(sizes, k);

    bad[first] = true;

    for (int i = 0; i < k; i++) {
        if (sizes[i] < n / k * min_deviation) {
            bad[i] = true;
        }
    }

    delete sizes;

    return bad;
}

int *
replace_medoids(int *M, int M_length, int *M_best, bool *bad, int *state_fixed, int state_length, int k, bool debug) {
//    int *M_random = shuffle(M, M_length);
    int *M_random;//= random_sample(M, k, M_length);

    if (debug) {
        M_random = not_random_sample(M, state_fixed, state_length, k, M_length);
    } else {
        M_random = random_sample(M, k, M_length);
    }

    int *M_current = new int[k];

    int j = 0;
    for (int i = 0; i < k; i++) {
        if (!bad[i]) {
            M_current[j] = M_best[i];
            j += 1;
        }
    }

    int old_count = j;
    int p = 0;
    while (j < k) {
        bool is_in = false;
        for (int i = 0; i < old_count; i++) {
            if (M_random[p] == M_current[i]) {
                is_in = true;
                break;
            }
        }
        if (!is_in) {
            M_current[j] = M_random[p];
            j += 1;
        }
        p += 1;
    }

//    delete M_random;

    return M_current;
}

void remove_outliers(at::Tensor data, int *C, bool **D, int *M, int n, int k, int d) {
    float *delta = new float[k];

    for (int i = 0; i < k; i++) {
        delta[i] = 1000000.;//todo not nice
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (i != j) {
                float msd = manhattan_segmental_distance(D[i], data, M[i], M[j], d);
                if (delta[i] > msd) {
                    delta[i] = msd;
                }
            }
        }
    }

    for (int p = 0; p < n; p++) {
        bool clustered = false;
        for (int i = 0; i < k; i++) {
            float msd = manhattan_segmental_distance(D[i], data, M[i], p, d);
            if (msd <= delta[i]) {
                clustered = true;
                break;
            }
        }
        if (!clustered) {
            C[p] = -1;
        }
    }
    delete delta;
}

std::vector <at::Tensor>
PROCLUS(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug) {

    /// Initialization Phase
    int n = data.size(0);
    int d = data.size(1);
    l = std::min(l, d);
    int Ak = std::min(n, int(a * k));
    int Bk = std::min(n, int(b * k));

    // Initialize fixed set of "random" integers for debugging
    int state_length = 1024;
    int *state_fixed;
    if (debug) {
        state_fixed = fill_with_indices(state_length);
    }

    int *indices = new int[n];
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    int *S;
    if (debug) {
        S = not_random_sample(indices, state_fixed, state_length, Ak, n);
    } else {
        S = random_sample(indices, Ak, n);
    }

    if (debug) {
        printf("S:\n");
        print_array(S, Ak);
    }

    float *dist = new float[n];
    float *new_dist = new float[n];
    int *M = new int[Bk];
    greedy(M, dist, new_dist, data, S, Bk, Ak, d);
    delete S;

    if (debug) {
        printf("M:\n");
        print_array(M, Bk);
    }

    // Iterative Phase
    float best_objective = std::numeric_limits<float>::max();
    int *M_current = new int[k];

    indices = new int[Bk];
    for (int i = 0; i < Bk; i++) {
        indices[i] = i;
    }
    int *M_random;
    if (debug) {
        M_random = not_random_sample(indices, state_fixed, state_length, k, Bk);
    } else {
        M_random = random_sample(indices, k, Bk);
    }

    for (int i = 0; i < k; i++) {
        int r_i = M_random[i];
        M_current[i] = M[r_i];
    }

    int termination_criterion = 0;
    int *M_best = M_current;
    int *C_best;
    bool *bad;

    int **L = new int *[k];
    for (int i = 0; i < k; i++) {
        L[i] = new int[n];
    }
    int *L_sizes = new int[k];

    while (termination_criterion < termination_rounds) {

        termination_criterion += 1;
        for (int i = 0; i < k; i++) {
            int m_i = M_current[i];
            compute_l2_norm_to_medoid(dist, data, M_current, m_i, k, d);
            dist[i] = std::numeric_limits<float>::max();
            float delta_i = *std::min_element(dist, dist + k);

            compute_l2_norm_to_medoid(dist, data, m_i, n, d);
            int j = 0;
            for (int p = 0; p < n; p++) {
                if (dist[p] < delta_i) {
                    L[i][j] = p;
                    j += 1;
                }
            }
            L_sizes[i] = j;

        }

        bool **D = find_dimensions(data, L, L_sizes, M_current, k, n, d, l);
        int *C = assign_points(data, D, M_current, n, d, k);

        float objective_function = evaluate_cluster(data, D, C, n, d, k);

        if (debug) {
//            printf("d_delta: ");
//            print_array_gpu(d_delta, k);
//            printf("d_termination_criterion: ");
//            print_array_gpu(d_termination_criterion, 1);
            std::vector <std::vector<int>> v_C;
            for (int i = 0; i < n; i++) {
                int c_id = C[i];
                if (c_id >= 0) {
                    while (v_C.size() <= c_id) {
                        std::vector<int> empty;
                        v_C.push_back(empty);
                    }
                    v_C[c_id].push_back(i);
                }
            }
            printf("C_sizes:");
            for (auto c: v_C) {
                printf(" %d", c.size());
            }
            printf("\n");

            printf("D: ");
            print_array(D, k, d);
            printf("cost: %f\n", objective_function);
            printf("M_current: ");
            print_array(M_current, k);
        }


        for (int i = 0; i < k; i++) {
            delete D[i];
        }
        delete D;
        if (objective_function < best_objective) {
            termination_criterion = 0;
            best_objective = objective_function;
            M_best = M_current;

            C_best = C;
            bad = bad_medoids(C, k, min_deviation, n);
        } else {
            delete C;
            delete M_current;
        }

        M_current = replace_medoids(M, Bk, M_best, bad, state_fixed, state_length, k, debug);

    }

    // Refinement Phase
    for (int i = 0; i < k; i++) {
        L_sizes[i] = 0;
    }

    for (int p = 0; p < n; p++) {
        L_sizes[C_best[p]] += 1;
    }

    int *l_j = zeros_1d<int>(k);
    for (int i = 0; i < n; i++) {
        int cl = C_best[i];
        L[cl][l_j[cl]] = i;
        l_j[cl] += 1;
    }
    delete C_best;

    bool **D = find_dimensions(data, L, L_sizes, M_best, k, n, d, l);

    int *C = assign_points(data, D, M_best, n, d, k);

    remove_outliers(data, C, D, M_best, n, k, d);

    std::vector <at::Tensor> r;

    torch::Tensor M_Tensor = torch::zeros({k}, torch::kInt32);
    for (int i = 0; i < k; i++) {
        M_Tensor[i] = M_best[i];
    }
    r.push_back(M_Tensor);

    torch::Tensor D_Tensor = torch::zeros({k, d}, torch::kBool);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            D_Tensor[i][j] = D[i][j];
        }
    }
    r.push_back(D_Tensor);

    torch::Tensor C_Tensor = torch::zeros({n}, torch::kInt32);
    for (int i = 0; i < n; i++) {
        C_Tensor[i] = C[i];
    }
    r.push_back(C_Tensor);


    for (int i = 0; i < k; i++) {
        delete L[i];
    }
    delete L;
    delete L_sizes;
    delete C;
    for (int i = 0; i < k; i++) {
        delete D[i];
    }
    delete D;
    delete M_best;
    delete M_current;
    delete M_random;
    delete dist;

    return r;
}