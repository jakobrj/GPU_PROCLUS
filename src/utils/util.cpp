#include <cstdio>
#include "util.h"


#define DEBUG false

float *compute_l2_norm_to_medoid(float **S, int m_i, int n, int d) {
    float *dist = new float[n];

    for (int i = 0; i < n; i++) {
        dist[i] = 0;
        for (int j = 0; j < d; j++) {
            float sub = S[i][j] - S[m_i][j];
            dist[i] += sub * sub;
        }
        dist[i] = std::sqrt(dist[i]);
    }

    return dist;
}


float *compute_l2_norm_to_medoid(at::Tensor data, int *S, int m_i, int n, int d) {
//    printf("compute_l2_norm_to_medoid before allocating, n=%d\n", n);
    float *dist = new float[n];
//    printf("compute_l2_norm_to_medoid after allocating\n");
    float *x_m_i = data[m_i].data_ptr<float>();
    for (int i = 0; i < n; i++) {
        float *x_S_i = data[S[i]].data_ptr<float>();
        dist[i] = 0;
        for (int j = 0; j < d; j++) {
            float x1 = x_S_i[j];
            float x2 = x_m_i[j];
            float sub = x1 - x2;
            dist[i] += sub * sub;
        }
        dist[i] = std::sqrt(dist[i]);
    }

    return dist;
}

void compute_l2_norm_to_medoid(float *dist, at::Tensor data, int *S, int m_i, int n, int d) {
//    printf("compute_l2_norm_to_medoid before allocating, n=%d\n", n);
//    float *dist = new float[n];
//    printf("compute_l2_norm_to_medoid after allocating\n");
    float *x_m_i = data[m_i].data_ptr<float>();
    for (int i = 0; i < n; i++) {
        float *x_S_i = data[S[i]].data_ptr<float>();
        dist[i] = 0;
        for (int j = 0; j < d; j++) {
            float x1 = x_S_i[j];
            float x2 = x_m_i[j];
            float sub = x1 - x2;
            dist[i] += sub * sub;
        }
        dist[i] = std::sqrt(dist[i]);
    }

//    return dist;
}


float *compute_l2_norm_to_medoid(at::Tensor data, int m_i, int n, int d) {
    float *dist = new float[n];

    float *x_m_i = data[m_i].data_ptr<float>();
    for (int i = 0; i < n; i++) {

        float *x_i = data[i].data_ptr<float>();
        dist[i] = 0;
        for (int j = 0; j < d; j++) {
            float sub = x_i[j] - x_m_i[j];
            dist[i] += sub * sub;
        }
        dist[i] = std::sqrt(dist[i]);
    }

    return dist;
}

void compute_l2_norm_to_medoid(float *dist, at::Tensor data, int m_i, int n, int d) {
    float *x_m_i = data[m_i].data_ptr<float>();
    for (int i = 0; i < n; i++) {

        float *x_i = data[i].data_ptr<float>();
        dist[i] = 0;
        for (int j = 0; j < d; j++) {
            float sub = x_i[j] - x_m_i[j];
            dist[i] += sub * sub;
        }
        dist[i] = std::sqrt(dist[i]);
    }
}

float *compute_l1_norm_to_medoid(float **S, int m_i, bool *D_i, int n, int d) {
    float *dist = new float[n];

    for (int i = 0; i < n; i++) {
        dist[i] = 0;
        int size = 0;
        for (int j = 0; j < d; j++) {
            if (D_i[j]) {
                dist[i] += std::abs(S[i][j] - S[m_i][j]);
                size++;
            }
        }

        dist[i] /= size;
    }

    return dist;
}


float *compute_l1_norm_to_medoid(at::Tensor data, int m_i, bool *D_i, int n, int d) {
    float *dist = new float[n];

    float *data_m_i = data[m_i].data_ptr<float>();

    for (int i = 0; i < n; i++) {
        dist[i] = 0;
        float *data_i = data[i].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            if (D_i[j]) {
                dist[i] += std::abs(data_i[j] - data_m_i[j]);
            }
        }
    }

    return dist;
}

int argmax_1d(float *values, int n) {
    int max_idx = -1;
    float max_value = -10000;//todo something smaller
    //printf("min: %f\n", max_value);
    for (int i = 0; i < n; i++) {
        if (values[i] >= max_value) {
            max_value = values[i];
            max_idx = i;
        }
    }
    return max_idx;
}

template<typename T>
int argmin_1d(T *values, int n) {
    int min_idx = -1;
    T min_value = std::numeric_limits<T>::max();
    for (int i = 0; i < n; i++) {
        if (values[i] < min_value) {
            min_value = values[i];
            min_idx = i;
        }
    }
    return min_idx;
}

template int argmin_1d<int>(int *values, int n);

template int argmin_1d<float>(float *values, int n);

std::pair<int, int> *argmin_2d(float **values, int n, int m) {
    int min_x = -1;
    int min_y = -1;
    float min_value = std::numeric_limits<float>::max();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (values[i][j] < min_value) {
                min_value = values[i][j];
                min_x = i;
                min_y = j;
            }
        }
    }
    return new std::pair<int, int>(min_x, min_y);
}

void index_wise_minimum(float *values_1, float *values_2, int n) {
    for (int i = 0; i < n; i++) {
        values_1[i] = std::min(values_1[i], values_2[i]);
    }
}

float mean_1d(float *values, int n) {
    float sum = 0.;
    for (int i = 0; i < n; i++) {
        sum += values[i];
    }
    return sum / (float) n;
}

bool all_close_1d(float *values_1, float *values_2, int n) {
    bool result = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(values_1[i] - values_2[i]) > 0.001)
            result = false;
    }
    return result;
}

bool all_close_2d(float **values_1, float **values_2, int n, int m) {
    bool result = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (std::abs(values_1[i][j] - values_2[i][j]) > 0.001)
                result = false;
        }
    }
    return result;
}

bool close(float value_1, float value_2) {
    return std::abs(value_1 - value_2) < 0.001;
}

int *shuffle(int *indices, int n) {

    //print_debug("shuffle - start\n", DEBUG);

    int *a = new int[n];
    for (int i = 0; i < n; i++) {
        a[i] = indices[i];
    }

    for (int i = n - 1; i > 0; i--) {
        int j = std::rand() % (i + 1);
        int tmp_idx = a[i];
        a[i] = a[j];
        a[j] = tmp_idx;
    }

    //print_debug("shuffle - end\n", DEBUG);
    return a;
}

int *random_sample(int *indices, int k, int n) {
    for (int i = 0; i < k; i++) {
        int j = std::rand() % n;
        int tmp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_idx;
    }
    return indices;
}

int *not_random_sample(int *in, int *state, int state_length, int k, int n) {
    for (int i = 0; i < k; i++) {
        int j = state[0] % n;//i % state_length
        state[0] += 11;

        int tmp_idx = in[i];
        in[i] = in[j];
        in[j] = tmp_idx;
    }
    return in;
}


template<typename T>
T **array_2d(int n, int m) {
    T **S = new T *[n];
    for (int i = 0; i < n; i++) {
        T *S_i = new T[m];
        S[i] = S_i;
    }
    return S;
}

template int **array_2d<int>(int n, int m);

template bool **array_2d<bool>(int n, int m);

template float **array_2d<float>(int n, int m);

template<typename T>
T **zeros_2d(int n, int m) {
    T **S = new T *[n];
    for (int i = 0; i < n; i++) {
        S[i] = new T[m];
        for (int j = 0; j < m; j++) {
            S[i][j] = 0;
        }
    }
    return S;
}

template int **zeros_2d<int>(int n, int m);

template bool **zeros_2d<bool>(int n, int m);

template float **zeros_2d<float>(int n, int m);

template<typename T>
T *zeros_1d(int n) {
    T *S = new T[n];
    for (int i = 0; i < n; i++) {
        S[i] = 0;
    }
    return S;
}

template int *zeros_1d<int>(int n);

template bool *zeros_1d<bool>(int n);

template float *zeros_1d<float>(int n);

float **gather_2d(float **S, int *indices, int k, int d) {
    float **R = array_2d<float>(k, d);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            R[i][j] = S[indices[i]][j];
        }
    }
    return R;
}


float **gather_2d(at::Tensor S, int *indices, int k, int d) {
    float **R = array_2d<float>(k, d);
    for (int i = 0; i < k; i++) {
        float *S_i = S[indices[i]].data_ptr<float>();
        for (int j = 0; j < d; j++) {
            R[i][j] = S_i[j];
        }
    }
    return R;
}

int *fill_with_indices(int n) {
    int *h_S = new int[n];
    for (int p = 0; p < n; p++) {
        h_S[p] = p;
    }
    return h_S;
}

void print_debug(char *str, bool debug) {
    if (debug)
        printf(str);
}

void print_array(float *x, int n) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%f ", (float) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    }
    printf("\n");
}

void print_array(int *x, int n) {
    int left = 300;
    int right = 300;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%d ", x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", x[i]);
        }
    }
    printf("\n");
}

void print_array(bool *x, int n) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%s ", x[i] ? "true" : "false");
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%s ", x[i] ? "true" : "false");
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%s ", x[i] ? "true" : "false");
        }
    }
    printf("\n");
}

void print_array(float **X, int n, int m) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            print_array(X[i], m);
        }
    } else {
        for (int i = 0; i < left; i++) {
            print_array(X[i], m);
        }
        printf(" ... \n");
        for (int i = n - right; i < n; i++) {
            print_array(X[i], m);
        }
    }
    printf("\n");
}

void print_array(bool **X, int n, int m) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            print_array(X[i], m);
        }
    } else {
        for (int i = 0; i < left; i++) {
            print_array(X[i], m);
        }
        printf(" ... \n");
        for (int i = n - right; i < n; i++) {
            print_array(X[i], m);
        }
    }
    printf("\n");
}