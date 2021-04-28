#include "PROCLUS.h"
#include "../utils/util.h"
#include "../utils/mem_util.h"
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

    int rnd_start = Ak / 2;//std::rand() % Ak
    M[0] = S[rnd_start];
    compute_l2_norm_to_medoid(dist, data, S, M[0], Ak, d);

    for (int i = 1; i < Bk; i++) {
        M[i] = S[argmax_1d(dist, Ak)];
        int m_i = M[i];

        compute_l2_norm_to_medoid(new_dist, data, S, m_i, Ak, d);
        index_wise_minimum(dist, new_dist, Ak);
    }
}

bool **find_dimensions(at::Tensor data, int **L, int *L_sizes, int *M, int k, int n, int d, int l) {

    float **X = array_2d<float>(k, d);
    float **Z = array_2d<float>(k, d);
    float *Y = array_1d<float>(k);
    bool **D = zeros_2d<bool>(k, d);
    float *sigma = array_1d<float>(k);


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
        for (int _ = 0; _ < 2; _++) {
            int j = argmin_1d<float>(Z[i], d);
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

    free(X, k);
    free(Z, k);
    free(Y);
    free(sigma);

    return D;
}

int *assign_points(at::Tensor data, bool **D, int *M, int n, int d, int k) {
    int *C = array_1d<int>(n);
    float **dist = array_2d<float>(n, k);
    for (int i = 0; i < k; i++) {

        float *tmp_dist = compute_l1_norm_to_medoid(data, M[i], D[i], n, d);
        for (int j = 0; j < n; j++) {
            dist[j][i] = tmp_dist[j];
        }
        free(tmp_dist);
    }

    for (int p = 0; p < n; p++) {
        int i = argmin_1d<float>(dist[p], k);
        C[p] = i;
    }

    for (int i = 0; i < k; i++) {
        C[M[i]] = i;
    }

    free(dist, n);

    return C;
}


float evaluate_cluster(at::Tensor data, bool **D, int *C, int n, int d, int k) {
    float **Y = zeros_2d<float>(k, d);
    float *w = array_1d<float>(k);

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

    free(Y, k);
    free(w);
    free(means, k);
    free(counts);

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

    free(sizes);

    return bad;
}

int *
replace_medoids(int *M, int M_length, int *M_best, bool *bad, int *M_random, int *state_fixed, int state_length, int k,
                bool debug) {
    int *M_kept = array_1d<int>(k);

    if (debug) {
        M_random = not_random_sample(M_random, state_fixed, state_length, k, M_length);
        //todo - this is why we dont get the same fixed random medoids
    } else {
        M_random = random_sample(M_random, k, M_length);
    }

    int *M_current = array_1d<int>(k);

    int j = 0;
    for (int i = 0; i < k; i++) {
        if (!bad[i]) {
            M_current[i] = M_best[i];
            M_kept[j] = M_best[i];
            j += 1;
        }
    }

    int old_count = j;
    int p = 0;
    for (int i = 0; i < k; i++) {
        if (bad[i]) {
            bool is_in = true;
            while (is_in) {
                is_in = false;
                for (int q = 0; q < old_count; q++) {
                    if (M[M_random[p]] == M_kept[q]) {
                        is_in = true;
                        p++;
                        break;
                    }
                }
            }
            M_current[i] = M[M_random[p]];
            M_kept[j] = M[M_random[p]];
            j += 1;
            p += 1;
        }
    }

    free(M_kept);

    return M_current;
}

void remove_outliers(at::Tensor data, int *C, bool **D, int *M, int n, int k, int d) {
    float *delta = array_1d<float>(k);

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
    free(delta);
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

    int *indices = array_1d<int>(n);
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

    float *dist = array_1d<float>(n);
    float *new_dist = array_1d<float>(n);
    int *M = array_1d<int>(Bk);
    greedy(M, dist, new_dist, data, S, Bk, Ak, d);
    free(S);
    free(new_dist);

    if (debug) {
        printf("M:\n");
        print_array(M, Bk);
    }

    // Iterative Phase
    float best_objective = std::numeric_limits<float>::max();
    int *M_current = array_1d<int>(k);

    indices = fill_with_indices(Bk);

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
    int *M_best = nullptr;
    int *C_best = nullptr;
    bool *bad = nullptr;

    int **L = array_2d<int>(k, n);
    int *L_sizes = array_1d<int>(k);

    while (termination_criterion < termination_rounds) {

        if (debug) {
            printf("\n\n------------\n");
        }

        for (int i = 0; i < k; i++) {
            int m_i = M_current[i];
            compute_l2_norm_to_medoid(dist, data, M_current, m_i, k, d);
            dist[i] = std::numeric_limits<float>::max();
            float delta_i = *std::min_element(dist, dist + k);

            compute_l2_norm_to_medoid(dist, data, m_i, n, d);
            int j = 0;
            for (int p = 0; p < n; p++) {
                if (dist[p] <= delta_i) {
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
            printf("L_sizes: ");
            print_array(L_sizes, k);
            printf("L:\n");
            print_array(L, k, n);
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
            printf("termination_criterion: %d\n", termination_criterion);
            printf("\n------------\n\n");
        }


        free(D, k);

        termination_criterion += 1;
        if (objective_function < best_objective) {
            termination_criterion = 0;
            best_objective = objective_function;
            if (M_best != nullptr) {
                free(M_best);
            }
            M_best = M_current;
            if (C_best != nullptr) {
                free(C_best);
            }
            C_best = C;
            if (bad != nullptr) {
                free(bad);
            }
            bad = bad_medoids(C, k, min_deviation, n);
        } else {
            free(C);
            free(M_current);
        }

        M_current = replace_medoids(M, Bk, M_best, bad, M_random, state_fixed, state_length, k, debug);
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

    free(bad);
    free(C);
    free(C_best);
    free(D, k);
    free(dist);
    free(L, k);
    free(l_j);
    free(L_sizes);
    free(M);
    free(M_best);
    free(M_current);
    free(M_random);
    if (debug) {
        free(state_fixed);
    }

    if (debug) {
        printf("missing deletions %d\n", get_allocated_count());
    }
    return r;
}


bool **
find_dimensions_KEEP(at::Tensor data, float **H, int *lambda, bool *bad, int **Delta_L, int *Delta_L_sizes,
                     int *L_sizes,
                     int *M, int k, int n, int d, int l) {
    //print_debug("find_dimensions - start\n", DEBUG);
    float **X = array_2d<float>(k, d);
    float **Z = array_2d<float>(k, d);
    float *Y = array_1d<float>(k);
    bool **D = zeros_2d<bool>(k, d);
    float *sigma = array_1d<float>(k);

    //print_debug("find_dimensions - init \n", DEBUG);

    //compute X,Y,Z

    for (int i = 0; i < k; i++) {
        if (bad[i]) {
            for (int j = 0; j < d; j++) {
                H[i][j] = 0.;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        float *x_m_i = data[M[i]].data_ptr<float>();
        for (int p = 0; p < Delta_L_sizes[i]; p++) {
            int point = Delta_L[i][p];
            float *x_p = data[point].data_ptr<float>();
            for (int j = 0; j < d; j++) {
                H[i][j] += lambda[i] * std::abs(x_p[j] - x_m_i[j]);
            }
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            if (L_sizes[i] == 0) {
                X[i][j] = 0;//todo should not happen!
            } else {
                X[i][j] = H[i][j] / L_sizes[i];
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
        for (int _ = 0; _ < 2; _++) {
            int j = argmin_1d<float>(Z[i], d);
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

    free(X, k);
    free(Y);
    free(Z, k);
    free(sigma);

    return D;
}

std::vector <at::Tensor>
PROCLUS_KEEP(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug) {

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

    int *indices = array_1d<int>(n);
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

    float *dist_tmp = array_1d<float>(n);
    float *new_dist_tmp = array_1d<float>(n);
    int *M = array_1d<int>(Bk);
    greedy(M, dist_tmp, new_dist_tmp, data, S, Bk, Ak, d);
    free(S);
    free(dist_tmp);
    free(new_dist_tmp);

    if (debug) {
        printf("M:\n");
        print_array(M, Bk);
    }

    // Iterative Phase
    float best_objective = std::numeric_limits<float>::max();
    int *M_current = array_1d<int>(k);

    indices = array_1d<int>(Bk);
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
    int *M_best = nullptr;
    int *C_best = nullptr;
    bool *bad = array_1d<bool>(k);
    for (int i = 0; i < k; i++) {
        bad[i] = true;
    }

    int **Delta_L = array_2d<int>(k, n);
    int **L = Delta_L;
    int *Delta_L_sizes = zeros_1d<int>(k);
    int *L_sizes = zeros_1d<int>(k);
    float **H = zeros_2d<float>(k, d);
    int *lambda = array_1d<int>(k);
    float *delta_prev = zeros_1d<float>(k);

    float **dist = zeros_2d<float>(k, n);
    while (termination_criterion < termination_rounds) {


        termination_criterion += 1;


        /// compute L
        for (int i = 0; i < k; i++) {
            if (bad[i]) {
                delta_prev[i] = -1.;
                L_sizes[i] = 0;
                int m_i = M_current[i];
                compute_l2_norm_to_medoid(dist[i], data, m_i, n, d);
            }
        }
        for (int i = 0; i < k; i++) {
            int m_i = M_current[i];
            float delta_i = std::numeric_limits<float>::max();
            for (int j = 0; j < k; j++) {
                int m_j = M_current[j];
                if (i != j) {
                    if (dist[i][m_j] < delta_i) {
                        delta_i = dist[i][m_j];
                    }
                }
            }

            int j = 0;
            for (int p = 0; p < n; p++) {
                if ((delta_i < dist[i][p] && dist[i][p] <= delta_prev[i])
                    || delta_prev[i] < dist[i][p] && dist[i][p] <= delta_i) {
                    Delta_L[i][j] = p;
                    j += 1;
                }
            }
            lambda[i] = delta_prev[i] < delta_i ? 1 : -1;
            delta_prev[i] = delta_i;
            Delta_L_sizes[i] = j;
            L_sizes[i] += lambda[i] * j;
        }

        bool **D = find_dimensions_KEEP(data, H, lambda, bad, Delta_L, Delta_L_sizes, L_sizes, M_current, k, n, d, l);
        int *C = assign_points(data, D, M_current, n, d, k);

        float objective_function = evaluate_cluster(data, D, C, n, d, k);

        if (debug) {
            printf("\n\n------------\n");
            printf("lambda:");
            print_array(lambda, k);
            printf("Delta_L_sizes: ");
            print_array(Delta_L_sizes, k);
            printf("L_sizes: ");
            print_array(L_sizes, k);
            printf("L:\n");
            print_array(Delta_L, k, n);
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
            printf("\n------------\n\n");
        }

        free(D, k);

        if (objective_function < best_objective) {
            termination_criterion = 0;
            best_objective = objective_function;
            if (M_best != nullptr) {
                free(M_best);
            }
            M_best = M_current;
            if (C_best != nullptr) {
                free(C_best);
            }
            C_best = C;
            if (bad != nullptr) {
                free(bad);
            }
            bad = bad_medoids(C, k, min_deviation, n);
        } else {
            free(C);
            free(M_current);
        }

        M_current = replace_medoids(M, Bk, M_best, bad, M_random, state_fixed, state_length, k, debug);


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

    free(bad);
    free(C);
    free(C_best);
    free(D, k);
    free(H, k);
    free(dist, k);
    free(L, k);//and Delta_L is the same
    free(Delta_L_sizes);
    free(L_sizes);
    free(l_j);
    free(lambda);
    free(delta_prev);
    free(M);
    free(M_best);
    free(M_current);
    free(M_random);

    if (debug) {
        free(state_fixed);
    }

    if (debug) {
        printf("missing deletions %d\n", get_allocated_count());
    }

    return r;
}


bool **
find_dimensions_SAVE(at::Tensor data, float **H, int *lambda, int *M_idx, int **Delta_L, int *Delta_L_sizes,
                     int *L_sizes, int *M_current, int k, int n, int d, int l) {

    float **X = array_2d<float>(k, d);
    float **Z = array_2d<float>(k, d);
    float *Y = array_1d<float>(k);
    bool **D = zeros_2d<bool>(k, d);
    float *sigma = array_1d<float>(k);


    //compute X,Y,Z
    for (int i = 0; i < k; i++) {
        float *x_m_i = data[M_current[i]].data_ptr<float>();
        for (int p = 0; p < Delta_L_sizes[i]; p++) {
            int point = Delta_L[i][p];
            float *x_p = data[point].data_ptr<float>();
            for (int j = 0; j < d; j++) {
                H[M_idx[i]][j] += lambda[i] * std::abs(x_p[j] - x_m_i[j]);
            }
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            X[i][j] = H[M_idx[i]][j] / L_sizes[M_idx[i]];
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
        for (int _ = 0; _ < 2; _++) {
            int j = argmin_1d<float>(Z[i], d);
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

    free(X, k);
    free(Y);
    free(Z, k);
    free(sigma);

    return D;
}


int *
replace_medoids_SAVE(int *M_idx, int *M_idx_best, int *M, int M_length, int *M_best, bool *bad, int *M_random,
                     int *state_fixed,
                     int state_length, int k,
                     bool debug) {
    int *M_kept = array_1d<int>(k);

    if (debug) {
        M_random = not_random_sample(M_random, state_fixed, state_length, k, M_length);
        //todo - this is why we dont get the same fixed random medoids
    } else {
        M_random = random_sample(M_random, k, M_length);
    }

    int *M_current = array_1d<int>(k);

    int j = 0;
    for (int i = 0; i < k; i++) {
        if (!bad[i]) {
            M_current[i] = M_best[i];
            M_idx[i] = M_idx_best[i];
            M_kept[j] = M_best[i];
            j += 1;
        }
    }

    int old_count = j;
    int p = 0;
    for (int i = 0; i < k; i++) {
        if (bad[i]) {
            bool is_in = true;
            while (is_in) {
                is_in = false;
                for (int q = 0; q < old_count; q++) {
                    if (M[M_random[p]] == M_kept[q]) {
                        is_in = true;
                        p++;
                        break;
                    }
                }
            }
            M_current[i] = M[M_random[p]];
            M_kept[j] = M[M_random[p]];
            M_idx[i] = M_random[p];
            j += 1;
            p += 1;
        }
    }
    free(M_kept);

    return M_current;
}


std::vector <at::Tensor>
PROCLUS_SAVE(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug) {

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

    int *indices = array_1d<int>(n);
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

    float *dist_tmp = array_1d<float>(n);
    float *new_dist_tmp = array_1d<float>(n);
    int *M = array_1d<int>(Bk);
    greedy(M, dist_tmp, new_dist_tmp, data, S, Bk, Ak, d);
    free(S);
    free(dist_tmp);
    free(new_dist_tmp);

    if (debug) {
        printf("M:\n");
        print_array(M, Bk);
    }

    // Iterative Phase
    float best_objective = std::numeric_limits<float>::max();
    int *M_current = array_1d<int>(k);

    indices = fill_with_indices(Bk);

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
    int *M_best = nullptr;
    int *C_best = nullptr;
    bool *bad = array_1d<bool>(k);
    for (int i = 0; i < k; i++) {
        bad[i] = true;
    }

    int **Delta_L = array_2d<int>(k, n);
    int **L = Delta_L;
    int *Delta_L_sizes = zeros_1d<int>(k);
    int *L_sizes = zeros_1d<int>(Bk);
    float **H = zeros_2d<float>(Bk, d);
    int *lambda = array_1d<int>(k);
    float *delta_prev = array_1d<float>(Bk);
    for (int i = 0; i < Bk; i++) {
        delta_prev[i] = -1.;
    }

    bool *dist_found = array_1d<bool>(Bk);
    for (int i = 0; i < Bk; i++) {
        dist_found[i] = false;
    }
    int *M_idx = array_1d<int>(k);
    int *M_idx_best = array_1d<int>(k);
    for (int i = 0; i < k; i++) {
        M_idx[i] = M_random[i];
        M_idx_best[i] = M_random[i];
    }
    float **dist = zeros_2d<float>(Bk, n);

    while (termination_criterion < termination_rounds) {

        termination_criterion += 1;

        /// compute L
        /// changed - start
        for (int i = 0; i < k; i++) {
            if (!dist_found[M_idx[i]]) {///todo change
                int m_i = M_current[i];
                compute_l2_norm_to_medoid(dist[M_idx[i]], data, m_i, n, d);
                dist_found[M_idx[i]] = true;
            }
        }
        for (int i = 0; i < k; i++) {
            int m_i = M_current[i];
            float delta_i = std::numeric_limits<float>::max();
            for (int j = 0; j < k; j++) {
                int m_j = M_current[j];
                if (i != j) {
                    if (dist[M_idx[i]][m_j] < delta_i) {
                        delta_i = dist[M_idx[i]][m_j];
                    }
                }
            }

            int j = 0;
            for (int p = 0; p < n; p++) {
                if ((delta_i < dist[M_idx[i]][p] && dist[M_idx[i]][p] <= delta_prev[M_idx[i]])
                    || delta_prev[M_idx[i]] < dist[M_idx[i]][p] && dist[M_idx[i]][p] <= delta_i) {
                    Delta_L[i][j] = p;
                    j += 1;
                }
            }
            lambda[i] = delta_prev[M_idx[i]] < delta_i ? 1 : -1;
            delta_prev[M_idx[i]] = delta_i;
            Delta_L_sizes[i] = j;
            L_sizes[M_idx[i]] += lambda[i] * j;
        }
        ///changed - end
        bool **D = find_dimensions_SAVE(data, H, lambda, M_idx, Delta_L, Delta_L_sizes, L_sizes, M_current, k, n, d, l);
        int *C = assign_points(data, D, M_current, n, d, k);

        float objective_function = evaluate_cluster(data, D, C, n, d, k);

        if (debug) {
            printf("\n\n------------\n");
            printf("lambda:");
            print_array(lambda, k);
            printf("Delta_L_sizes: ");
            print_array(Delta_L_sizes, k);
            printf("L_sizes: ");
            print_array(L_sizes, Bk);
            printf("L:\n");
            print_array(Delta_L, k, n);

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
            printf("\n------------\n\n");
        }


        free(D, k);
        if (objective_function < best_objective) {
            termination_criterion = 0;
            best_objective = objective_function;
            if (M_best != nullptr) {
                free(M_best);
            }
            M_best = M_current;
            for (int i = 0; i < k; i++) {
                M_idx_best[i] = M_idx[i];
            }
            if (C_best != nullptr) {
                free(C_best);
            }
            C_best = C;
            if (bad != nullptr) {
                free(bad);
            }
            bad = bad_medoids(C, k, min_deviation, n);
        } else {
            free(C);
            free(M_current);
        }

        M_current = replace_medoids_SAVE(M_idx, M_idx_best, M, Bk, M_best, bad, M_random, state_fixed, state_length, k,
                                         debug);


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

    free(bad);
    free(C);
    free(C_best);
    free(D, k);
    free(H, Bk);
    free(dist, Bk);
    free(dist_found);
    free(L, k);
    free(Delta_L_sizes);
    free(L_sizes);
    free(l_j);
    free(lambda);
    free(delta_prev);
    free(M);
    free(M_idx_best);
    free(M_idx);
    free(M_best);
    free(M_current);
    free(M_random);

    if (debug) {
        free(state_fixed);
    }

    if (debug) {
        printf("missing deletions %d\n", get_allocated_count());
    }

    return r;
}


std::vector <std::vector<at::Tensor>>
PROCLUS_PARAM(at::Tensor data, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
              int termination_rounds, bool debug) {
    int k_max = ks[0];
    int l_max = ls[0];

    /// Initialization Phase
    int n = data.size(0);
    int d = data.size(1);
    int Ak = std::min(n, int(a * k_max));
    int Bk = std::min(n, int(b * k_max));


    // Initialize fixed set of "random" integers for debugging
    int state_length = 1024;
    int *state_fixed;
    if (debug) {
        state_fixed = fill_with_indices(state_length);
    }

    int *indices = fill_with_indices(n);

    int *S;//S is the same as indices
    if (debug) {
        S = not_random_sample(indices, state_fixed, state_length, Ak, n);
    } else {
        S = random_sample(indices, Ak, n);
    }


    if (debug) {
        printf("S:\n");
        print_array(S, Ak);
    }

    float *dist_tmp = array_1d<float>(n);
    float *new_dist_tmp = array_1d<float>(n);
    int *M = array_1d<int>(Bk);
    greedy(M, dist_tmp, new_dist_tmp, data, S, Bk, Ak, d);
    free(S);
    free(dist_tmp);
    free(new_dist_tmp);

    if (debug) {
        printf("M:\n");
        print_array(M, Bk);
    }

    // Iterative Phase
    float best_objective = std::numeric_limits<float>::max();
    int *M_current = array_1d<int>(k_max);

    indices = fill_with_indices(Bk);
    int *M_random;
    if (debug) {
        M_random = not_random_sample(indices, state_fixed, state_length, k_max, Bk);
    } else {
        M_random = random_sample(indices, k_max, Bk);
    }

    for (int i = 0; i < k_max; i++) {
        int r_i = M_random[i];
        M_current[i] = M[r_i];
    }

    int termination_criterion = 0;
    int *M_best = nullptr;
    int *C_best = nullptr;
    bool *bad = array_1d<bool>(k_max);
    for (int i = 0; i < k_max; i++) {
        bad[i] = true;
    }

    int **Delta_L = array_2d<int>(k_max, n);
    int *Delta_L_sizes = zeros_1d<int>(k_max);
    int *L_sizes = zeros_1d<int>(Bk);
    float **H = zeros_2d<float>(Bk, d);
    int *lambda = array_1d<int>(k_max);
    float *delta_prev = array_1d<float>(Bk);
    for (int i = 0; i < Bk; i++) {
        delta_prev[i] = -1.;
    }

    int **r_L = array_2d<int>(k_max, n);
    int *r_L_sizes = zeros_1d<int>(k_max);


    bool *dist_found = array_1d<bool>(Bk);
    for (int i = 0; i < Bk; i++) {
        dist_found[i] = false;
    }
    int *M_idx = array_1d<int>(k_max);
    int *M_idx_best = array_1d<int>(k_max);
    for (int i = 0; i < k_max; i++) {
        M_idx[i] = M_random[i];
        M_idx_best[i] = M_random[i];
    }
    float **dist = zeros_2d<float>(Bk, n);
    if (debug) {
        printf("before checking all params\n");
    }
    std::vector <std::vector<at::Tensor>> R;
    for (int k_idx = 0; k_idx < ks.size(); k_idx++) {
        int k = ks[k_idx];
        for (int l_idx = 0; l_idx < ls.size(); l_idx++) {
            int l = ls[l_idx];
            l = std::min(l, d);

            if (l_idx == 0 & k_idx == 0) {

            } else {
                for (int i = 0; i < k; i++) {
                    M_current[i] = M_best[i];
                    M_idx[i] = M_idx_best[i];
                }
            }
            termination_criterion = 0;
            best_objective = std::numeric_limits<float>::max();

            if (debug) {
                printf("before iterative phase\n");
            }

            while (termination_criterion < termination_rounds) {


                termination_criterion += 1;


                /// compute L

                if (debug) {
                    printf("compute L\n");
                }

                for (int i = 0; i < k; i++) {
                    if (!dist_found[M_idx[i]]) {
                        int m_i = M_current[i];
                        compute_l2_norm_to_medoid(dist[M_idx[i]], data, m_i, n, d);
                        dist_found[M_idx[i]] = true;
                    }
                }
                for (int i = 0; i < k; i++) {
                    int m_i = M_current[i];
                    float delta_i = std::numeric_limits<float>::max();
                    for (int j = 0; j < k; j++) {
                        int m_j = M_current[j];
                        if (i != j) {
                            if (dist[M_idx[i]][m_j] < delta_i) {
                                delta_i = dist[M_idx[i]][m_j];
                            }
                        }
                    }

                    int j = 0;
                    for (int p = 0; p < n; p++) {
                        if ((delta_i < dist[M_idx[i]][p] && dist[M_idx[i]][p] <= delta_prev[M_idx[i]])
                            || delta_prev[M_idx[i]] < dist[M_idx[i]][p] && dist[M_idx[i]][p] <= delta_i) {
                            Delta_L[i][j] = p;
                            j += 1;
                        }
                    }
                    lambda[i] = delta_prev[M_idx[i]] < delta_i ? 1 : -1;
                    delta_prev[M_idx[i]] = delta_i;
                    Delta_L_sizes[i] = j;
                    L_sizes[M_idx[i]] += lambda[i] * j;
                }

                if (debug) {
                    printf("\n\n------------\n");
                    printf("find_dimensions_SAVE\n");
                    printf("Delta_L_sizes:");
                    print_array(Delta_L_sizes, k);
                    printf("L_sizes:");
                    print_array(L_sizes, M_idx, k);
                    printf("l: %d, k: %d\n", l, k);
                }
                bool **D = find_dimensions_SAVE(data, H, lambda, M_idx, Delta_L, Delta_L_sizes, L_sizes, M_current, k,
                                                n, d, l);

                if (debug) {
                    printf("assign_points\n");
                }
                int *C = assign_points(data, D, M_current, n, d, k);
                if (debug) {
                    printf("evaluate_cluster\n");
                }
                float objective_function = evaluate_cluster(data, D, C, n, d, k);

                if (debug) {
                    printf("lambda:");
                    print_array(lambda, k);
                    printf("Delta_L_sizes: ");
                    print_array(Delta_L_sizes, k);
                    printf("L_sizes: ");
                    print_array(L_sizes, Bk);
                    printf("L:\n");
                    print_array(Delta_L, k, n);

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
                    printf("\n------------\n\n");
                }

                free(D, k);

                if (objective_function < best_objective) {
                    termination_criterion = 0;
                    best_objective = objective_function;
                    if (M_best != nullptr) {
                        free(M_best);
                    }
                    M_best = M_current;
                    for (int i = 0; i < k; i++) {
                        M_idx_best[i] = M_idx[i];
                    }
                    if (C_best != nullptr) {
                        free(C_best);
                    }
                    C_best = C;
                    free(bad);
                    bad = bad_medoids(C, k, min_deviation, n);
                } else {
                    free(C);
                    free(M_current);
                }

                M_current = replace_medoids_SAVE(M_idx, M_idx_best, M, Bk, M_best, bad, M_random, state_fixed,
                                                 state_length, k, debug);


            }


            // Refinement Phase
            for (int i = 0; i < k; i++) {
                r_L_sizes[i] = 0;
            }

            for (int p = 0; p < n; p++) {
                r_L_sizes[C_best[p]] += 1;
            }

            int *l_j = zeros_1d<int>(k);
            for (int i = 0; i < n; i++) {
                int cl = C_best[i];
                r_L[cl][l_j[cl]] = i;
                l_j[cl] += 1;
            }

            bool **D = find_dimensions(data, r_L, r_L_sizes, M_best, k, n, d, l);

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

            R.push_back(r);
            free(C);
            free(D, k);
            free(l_j);
        }
    }

    free(bad);
    free(C_best);
    free(H, Bk);
    free(dist, Bk);
    free(dist_found);
    free(Delta_L, k_max);
    free(Delta_L_sizes);
    free(L_sizes);
    free(r_L, k_max);
    free(r_L_sizes);
    free(lambda);
    free(delta_prev);
    free(M);
    free(M_idx_best);
    free(M_idx);
    free(M_best);
    free(M_current);
    free(M_random);

    if (debug) {
        free(state_fixed);
    }

    if (debug) {
        printf("missing deletions %d\n", get_allocated_count());
    }

    return R;
}