//
// Created by jakobrj on 3/8/21.
//

#include "../utils/util.h"
#include "../utils/gpu_util.cuh"
#include "../utils/cuda_util.cuh"
#include "GPU_PROCLUS.cuh"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_SMALL 128

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ __forceinline__

float atomicMin(float *addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *) addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((unsigned int *) addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__

float atomicMax(float *addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *) addr, __float_as_int(value))) :
          __uint_as_float(atomicMin((unsigned int *) addr, __float_as_uint(value)));

    return old;
}

__global__
void gpu_greedy_kernel_dist_max(float *d_max_value, float *d_data, int *M, int *d_S, float *dist, int Ak, int d,
                                int mediod_idx) {
    int m_i = M[mediod_idx];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Ak; i += blockDim.x * gridDim.x) {
        float distance = 0;
        for (int j = 0; j < d; j++) {
            float sub = d_data[d_S[i] * d + j] - d_data[m_i * d + j];
            distance += sub * sub;
        }
        dist[i] = sqrt(distance);
    }

    __shared__ float max_value;
    max_value = 0.;
    __syncthreads();

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < Ak; v += blockDim.x * gridDim.x) {
        atomicMax(&max_value, dist[v]);//todo does this work?
    }

    __syncthreads();
    if (threadIdx.x == 0)
        atomicMax(&d_max_value[0], max_value);
}

__global__
void gpu_greedy_kernel_largest_2(float *d_max_value, int *d_S, int *M, float *dist, int *d_prev, int Ak, int i, int n) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < Ak; v += blockDim.x * gridDim.x) {
        if (dist[v] == d_max_value[0]) {
            M[i] = d_S[v];
            d_prev[0] = v;
        }
    }
}

__global__
void gpu_greedy_kernel_dist_min_max(float *d_max_value, float *d_data, int *M, int *d_S, float *dist, int Ak, int d,
                                    int mediod_idx) {

    __shared__ float max_value;
    max_value = 0.;
    __syncthreads();

    int m_i = M[mediod_idx];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Ak; i += blockDim.x * gridDim.x) {
        float distance = 0;
        for (int j = 0; j < d; j++) {
            float sub = d_data[d_S[i] * d + j] - d_data[m_i * d + j];
            distance += sub * sub;
        }

        distance = sqrt(distance);
        //finding the min distance here instead of in two kernels reduced the running time by 4ms out of 20ms
        if (distance < dist[i]) {
            dist[i] = distance;
        }

        atomicMax(&max_value, dist[i]);//first find local (within block)
    }


    __syncthreads();
    if (threadIdx.x == 0)
        atomicMax(&d_max_value[0], max_value);//then find global
}

int *gpu_greedy(float *d_data, int *d_S, int Bk, int Ak, int d, int n) {

    //allocate result
    int *d_M;
    cudaMalloc(&d_M, Bk * sizeof(int));

    //allocate tmp
    float *d_dist;
    cudaMalloc(&d_dist, Ak * sizeof(float));
    int *d_prev;
    cudaMalloc(&d_prev, sizeof(int));
    cudaMemset(d_prev, 0, sizeof(int));
    float *d_max_value;
    cudaMalloc(&d_max_value, sizeof(float));
    cudaMemset(d_max_value, 0, sizeof(float));

//    int rnd_start = std::rand() % Ak;
    int rnd_start = Ak / 2;

    set(d_M, d_S, 0, rnd_start);
    int number_of_blocks = Ak / BLOCK_SIZE_SMALL;
    if (Ak % BLOCK_SIZE_SMALL) number_of_blocks++;
    dim3 grid(number_of_blocks);

    gpu_greedy_kernel_dist_max << < grid, BLOCK_SIZE_SMALL >> > (d_max_value, d_data, d_M, d_S, d_dist, Ak, d, 0);

    for (int i = 1; i < Bk; i++) {

        gpu_greedy_kernel_largest_2 << < grid, BLOCK_SIZE_SMALL >> > (d_max_value, d_S, d_M, d_dist, d_prev, Ak, i, n);

        cudaMemset(d_max_value, 0, sizeof(float));

        gpu_greedy_kernel_dist_min_max << < grid, BLOCK_SIZE_SMALL >> >
                                                  (d_max_value, d_data, d_M, d_S, d_dist, Ak, d, i);
    }


    //free tmp
    cudaFree(d_dist);
    cudaFree(d_prev);
    cudaFree(d_max_value);

    return d_M;
}

__global__
void gpu_compute_L_kernel_sum_dist_V2(float *d_dist_n_k, int *d_M_current, float *d_data, int n, int d, int k) {
    int i = blockIdx.x;
    int m_i = d_M_current[i];

    extern __shared__ float s_data_i[];

    if (threadIdx.x < d) {
        s_data_i[threadIdx.x] = d_data[m_i * d + threadIdx.x];
    }

    __syncthreads();

    for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < n; p += gridDim.y * blockDim.x) {//independent
        float sum = 0;
        for (int j = 0; j < d; j++) {//we have plenty to parallelize over here - so we can avoid the atomic
            float sub = d_data[p * d + j] - s_data_i[j];
            sum += sub * sub;
        }
        d_dist_n_k[i * n + p] = std::sqrt(sum);
    }
}

__global__
void gpu_compute_L_kernel_compute_delta_V2(float *d_delta, float *d_dist_n_k, int *d_M_current, int n, int k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {//independent
        d_delta[i] = 1000000.;//todo not nice
        for (int j = 0; j < k; j++) {
            int p = d_M_current[j];
            if (i != j) {
                if (d_dist_n_k[i * n + p] <= d_delta[i]) {
                    d_delta[i] = d_dist_n_k[i * n + p];
                }
            }
        }
    }
}

__global__
void gpu_compute_L_kernel_compute_L_V2(int *d_L, int *d_L_sizes, float *d_delta, float *d_dist_n_k, int n, int k) {
    for (int i = blockIdx.x; i < k; i += gridDim.x) {//independent
        for (int p = threadIdx.x; p < n; p += blockDim.x) {
            if (d_dist_n_k[i * n + p] <= d_delta[i]) {
                int old_size = atomicInc((unsigned int *) &d_L_sizes[i], n);
                d_L[i * n + old_size] = p;
            }
        }
    }
}

void gpu_compute_L(int *d_L, int *d_L_sizes,
                   float *d_dist_n_k,
                   float *d_delta,
                   int *d_M_current,
                   float *d_data,
                   int n, int d, int k) {

    int number_of_blocks = n / BLOCK_SIZE_SMALL;
    if (n % BLOCK_SIZE_SMALL) number_of_blocks++;
    dim3 grid_k_n(k, number_of_blocks);
    gpu_compute_L_kernel_sum_dist_V2 << < grid_k_n, min(n, BLOCK_SIZE_SMALL), d * sizeof(float) >> >
                                                                              (d_dist_n_k, d_M_current,
                                                                                      d_data,
                                                                                      n, d, k);

    //compute delta
    gpu_compute_L_kernel_compute_delta_V2 << < 1, k >> > (d_delta, d_dist_n_k, d_M_current, n, k);

    //compute L
    cudaMemset(d_L_sizes, 0, k * sizeof(int));
    gpu_compute_L_kernel_compute_L_V2 << < k, min(n, BLOCK_SIZE) >> > (d_L, d_L_sizes, d_delta, d_dist_n_k, n, k);
}

__global__
void
gpu_find_dimensions_kernel_Z(float *__restrict__ d_Z, const float *__restrict__ d_X, const int k, const int d) {

    int i = blockIdx.x;//independent for different k
    int j = threadIdx.x;//independent for different d

    __shared__ float Y_i;
    Y_i = 0.;
    __shared__ float sigma_i;
    sigma_i = 0.;
    __syncthreads();

    float X_ij = d_X[i * d + j];
    atomicAdd(&Y_i, X_ij / d);
    __syncthreads();
////
    float sub = X_ij - Y_i;
    atomicAdd(&sigma_i, sub * sub);
    __syncthreads();
    if (threadIdx.x == 0) {//only one should do this
        sigma_i /= (d - 1);
        sigma_i = std::sqrt(sigma_i);
    }
    __syncthreads();
////
    d_Z[i * d + j] = sub / sigma_i;
}

__global__
void gpu_find_dimensions_kernel_X(float *d_X,
                                  int *d_L, int *d_L_sizes,
                                  int *d_M_current,
                                  float *d_data,
                                  int n, int d, int k) {
    int i = blockIdx.x; //independent for different k
    int j = threadIdx.x; //independent for different d

    int m_i = d_M_current[i];
    int L_i_sizes = d_L_sizes[i];
    float data_ij = d_data[m_i * d + j];

    float sum = 0.;
    for (int p = blockDim.y * blockIdx.y + threadIdx.y; p < L_i_sizes; p += gridDim.y * blockDim.y) {
        int point = d_L[i * n + p];
        sum += std::abs(d_data[point * d + j] - data_ij);
    }

    atomicAdd(&d_X[i * d + j], sum / L_i_sizes);

}


__global__
void gpu_find_dimensions_kernel_X_v2(float *d_X,
                                     int *d_L, int *d_L_sizes,
                                     int *d_M_current,
                                     float *d_data,
                                     int n, int d, int k) {
    int i = blockIdx.x; //independent for different k
    int j = blockIdx.y; //independent for different d

    int m_i = d_M_current[i];
    int L_i_sizes = d_L_sizes[i];
    float data_ij = d_data[m_i * d + j];

    float sum = 0.;
    for (int p = threadIdx.x; p < L_i_sizes; p += blockDim.x) {
        int point = d_L[i * n + p];
        sum += std::abs(d_data[point * d + j] - data_ij);
    }

    atomicAdd(&d_X[i * d + j], sum / L_i_sizes);

}


__global__
void gpu_find_dimensions_kernel_compute_D(bool *d_D, float *d_Z, int k, int d, int l) {
    //# ensuring that we find atleast 2 for each and than the k*l #todo fast - sort first instead

    extern __shared__ float min_values[];
    int *i_was_firsts = (int *) &min_values[k];
    __shared__ float min_value;
    __shared__ int i_was_first;

    for (int _ = 0; _ < 2; _++) {

        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            min_values[i] = 1000000.;//todo not nice
            i_was_firsts[i] = 1;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = threadIdx.y; j < d; j += blockDim.y) {
                atomicMin(&min_values[i], d_Z[i * d + j]);
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = threadIdx.y; j < d; j += blockDim.y) {
                if (d_Z[i * d + j] == min_values[i]) {
                    int was_i_first = atomicCAS(&i_was_firsts[i], 1, 0); //(old == compare ? val : old)
                    if (was_i_first) {
                        d_Z[i * d + j] = 1000000.;//todo not nice
                        d_D[i * d + j] = true;
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int _ = k * 2; _ < k * l; _++) {
        min_value = 1000000.;//todo not nice
        i_was_first = 1;
        __syncthreads();

        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = threadIdx.y; j < d; j += blockDim.y) {
                atomicMin(&min_value, d_Z[i * d + j]);
            }
        }

        __syncthreads();
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = threadIdx.y; j < d; j += blockDim.y) {
                if (d_Z[i * d + j] == min_value) {
                    int was_i_first = atomicCAS(&i_was_first, 1, 0); //(old == compare ? val : old)
                    if (was_i_first) {
                        d_Z[i * d + j] = 1000000.;//todo not nice
                        d_D[i * d + j] = true;
                    }
                }
            }
        }
        __syncthreads();
    }
}


__global__
void gpu_find_dimensions_kernel_D(bool *d_D, float *d_Z, int k, int d, int l) {
    //# ensuring that we find atleast 2 for each and than the k*l #todo fast - sort first instead

    for (int _ = 0; _ < 2; _++) {
        for (int i = 0; i < k; i++) {
            float min_value = 100000.;
            int best_j = 0;
            for (int j = 0; j < d; j++) {
                if (d_Z[i * d + j] < min_value) {
                    min_value = d_Z[i * d + j];
                    best_j = j;;
                }
            }
            d_Z[i * d + best_j] = 1000000.;//todo not nice
            d_D[i * d + best_j] = true;
        }
    }

    for (int _ = k * 2; _ < k * l; _++) {
        float min_value = 100000.;
        int best_i = 0;
        int best_j = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                if (d_Z[i * d + j] < min_value) {
                    min_value = d_Z[i * d + j];
                    best_j = j;
                    best_i = i;
                }
            }
        }
        d_Z[best_i * d + best_j] = 1000000.;//todo not nice
        d_D[best_i * d + best_j] = true;
    }
}

void gpu_find_dimensions(bool *d_D, float *d_Z, float *d_X,
                         int *d_L, int *d_L_sizes,
                         int *d_M_current,
                         float *d_data,
                         int n, int d, int k, int l) {
    int number_of_blocks = (k * d) / BLOCK_SIZE;
    if ((k * d) % BLOCK_SIZE) number_of_blocks++;

    set_all << < number_of_blocks, min(k * d, BLOCK_SIZE) >> > (d_X, 0, k * d);


    int remaining_d = BLOCK_SIZE / d;
    int number_of_blocks_X_join_v2 = (n / k) / remaining_d;
    if ((n / k) % remaining_d) number_of_blocks_X_join_v2++;
    dim3 grid_X_join_v2(k, number_of_blocks_X_join_v2);
    dim3 block_X_join_v2(d, remaining_d);
//    dim3 grid_X_join_v2(k, d);
//    dim3
//    block_X_join_v2(BLOCK_SIZE);

    cudaMemset(d_X, 0, d * k * sizeof(float));
    gpu_find_dimensions_kernel_X << < grid_X_join_v2, block_X_join_v2 >> > (d_X,
            d_L, d_L_sizes,
            d_M_current,
            d_data,
            n, d, k);
//    gpu_find_dimensions_kernel_X_v2 << < grid_X_join_v2, block_X_join_v2 >> > (d_X,
//            d_L, d_L_sizes,
//            d_M_current,
//            d_data,
//            n, d, k);

    gpu_find_dimensions_kernel_Z << < k, d >> > (d_Z, d_X, k, d);

    //compute D
    set_all << < number_of_blocks, min(k * d, BLOCK_SIZE) >> > (d_D, false, k * d);
    dim3 block(min(32, k), min(32, d));
    gpu_find_dimensions_kernel_compute_D << < 1, block, 2 * k * sizeof(float) >> > (d_D, d_Z, k, d, l);
}

__global__
void
gpu_restructure_D(int *__restrict__ d_Ds, int *__restrict__ d_D_sizes, const bool *__restrict__ d_D, int d, int k) {

    int i = blockIdx.x;
    int j = threadIdx.x;

    if (d_D[i * d + j]) {
        int idx = atomicInc((unsigned int *) &d_D_sizes[i], d);
        d_Ds[i * d + idx] = j;
    }
}

//__global__
//void
//gpu_assign_points_kernel(int *__restrict__ d_Ds, int *__restrict__ d_D_sizes,
//                         int *__restrict__ d_C, int *__restrict__ d_C_size,
//                         const float *__restrict__ d_data, const int *__restrict__ d_M_current,
//                         const int n, const int k, const int d) {
//
//    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += blockDim.x * gridDim.x) {
//        float min_value = 1000000.;
//        int best_i = 0;
//        for (int i = 0; i < k; i++) {
//            int m_i = d_M_current[i];
//            int size = d_D_sizes[i];
//
//            float dist = 0;
//            for (int l = 0; l < size; l++) {
//                int j = d_Ds[i * d + l];
//                dist += abs(d_data[p * d + j] - d_data[m_i * d + j]);
//            }
//            dist /= size;
//
//            if (dist < min_value) {
//                min_value = dist;
//                best_i = i;
//            }
//        }
//        int idx = atomicInc((unsigned int *) &d_C_size[best_i], n);
//        d_C[best_i * n + idx] = p;
//
//    }
//}
__global__
void
gpu_assign_points_kernel(int *__restrict__ d_Ds, int *__restrict__ d_D_sizes,
                         int *__restrict__ d_C, int *__restrict__ d_C_size,
                         const float *__restrict__ d_data, const int *__restrict__ d_M_current,
                         const int n, const int k, const int d) {

    extern __shared__ float s_min_value[];

    float dist = 0;

    int i = threadIdx.y;
    int m_i = d_M_current[i];
    int size = d_D_sizes[i];

    int p = blockIdx.x * blockDim.x + threadIdx.x;

    s_min_value[threadIdx.x] = 1000000.;
    __syncthreads();

    if (p < n) {
        dist = 0;

        for (int l = 0; l < size; l++) {
            int j = d_Ds[i * d + l];
            dist += abs(d_data[p * d + j] - d_data[m_i * d + j]);
        }

        dist /= size;

        atomicMin(&s_min_value[threadIdx.x], dist);
    }

    __syncthreads();

    if (p < n) {
        if (dist == s_min_value[threadIdx.x]) {
            int idx = atomicInc((unsigned int *) &d_C_size[i], n);
            d_C[i * n + idx] = p;
        }
    }
}

void gpu_assign_points(int *d_C, int *d_C_sizes,
                       bool *d_D, int *d_Ds, int *d_D_sizes,
                       int *d_M_current,
                       float *d_data,
                       int n, int d, int k) {

    int remaining = BLOCK_SIZE_SMALL / k;
    int number_of_blocks = n / remaining;
    if (n % remaining) number_of_blocks++;
    dim3 block_n_k(min(n, remaining), k);

//    int number_of_blocks = n / BLOCK_SIZE;
//    if (n % BLOCK_SIZE) number_of_blocks++;

    cudaMemset(d_C_sizes, 0, k * sizeof(float));
    cudaMemset(d_Ds, 0, k * d * sizeof(int));
    cudaMemset(d_D_sizes, 0, k * sizeof(int));

    gpu_restructure_D << < k, d >> > (d_Ds, d_D_sizes, d_D, d, k);

    gpu_assign_points_kernel << < number_of_blocks, block_n_k, min(n, remaining) * sizeof(float) >> > (
            d_Ds, d_D_sizes, d_C, d_C_sizes, d_data, d_M_current, n, k, d);

//    gpu_assign_points_kernel<<<number_of_blocks, BLOCK_SIZE>>>(d_Ds, d_D_sizes, d_C, d_C_sizes, d_data, d_M_current, n,
//                                                               k, d);
}


//todo we should reconsider how we represent clustering - should we represent it in to different way? or just change it at the very end?
__global__
void gpu_evaluate_cluster_kernel(float *d_cost, int *d_C,
                                 int *d_C_size, bool *d_D, int *d_D_sizes,
                                 float *d_data,
                                 int n, int d, int k) { //  --  40.99%  413ms

    __shared__ float tmp_mean;
    __shared__ float tmp_cost;
    float tmp;

    int j = blockIdx.x;//j is the dimension within d dimensions
    int i = blockIdx.y;//i is the mediod / cluster within k clusters

    int size = d_C_size[i];
    int tmp_2 = d_D_sizes[i] * n;

    if (d_D[i * d + j]) {
        tmp = 0.;
        tmp_mean = 0.;
        __syncthreads();
        for (int l = threadIdx.x; l < size; l += blockDim.x) {
            int p = d_C[i * n + l];
            tmp += d_data[p * d + j];
        }
        atomicAdd(&tmp_mean, tmp / size);

        tmp_cost = 0;
        __syncthreads();
        tmp = 0.;
        for (int l = threadIdx.x; l < size; l += blockDim.x) {
            int p = d_C[i * n + l];
            tmp += abs(d_data[p * d + j] - tmp_mean);
        }

        atomicAdd(&tmp_cost, tmp / tmp_2);
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(&d_cost[0], tmp_cost);
    }
}

void
gpu_evaluate_cluster(float *d_cost, int *d_C, int *d_C_sizes, bool *d_D, int *d_D_sizes, float *d_data,
                     int n, int d, int k) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    dim3 grid(d, k);
    dim3 block(min(BLOCK_SIZE, min((int) cuda_cores / k, (int) n / k)));

    cudaMemset(d_cost, 0, sizeof(float));
    gpu_evaluate_cluster_kernel << < grid, block >> > (d_cost, d_C, d_C_sizes, d_D, d_D_sizes, d_data,
            n, d, k);
}

__global__
void
gpu_update_best_kernel_is_best(float *d_objective_function, float *d_best_objective, int *d_termination_criterion) {
    d_termination_criterion[0]++;
    if (d_objective_function[0] < d_best_objective[0]) {
        d_termination_criterion[0] = 0;
        d_best_objective[0] = d_objective_function[0];
    }
}

__global__
void
gpu_update_best_kernel_init_k(int *d_termination_criterion, int *d_M_best, int *d_M_current,
                              bool *d_bad, int k) {

    if (d_termination_criterion[0] == 0) {//todo worng!!!! then we allways pick the last????
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            d_M_best[i] = d_M_current[i];
            d_bad[i] = false;
        }
    }
}

__global__
void
gpu_update_best_kernel_C(int *d_C_best, int *d_C_sizes_best, int *d_C, int *d_C_sizes, int *d_termination_criterion,
                         int n) {

    if (d_termination_criterion[0] == 0) {
        int i = blockIdx.x;

        int C_i_size = d_C_sizes[i];
        for (int p_id = threadIdx.x; p_id < C_i_size; p_id += blockDim.x) {
            d_C_best[i * n + p_id] = d_C[i * n + p_id];
        }

        if (threadIdx.x == 0) {
            d_C_sizes_best[i] = d_C_sizes[i];
        }
    }
}

__global__
void gpu_update_best_kernel_find_bad(int *d_C_sizes_best, int *d_termination_criterion, bool *d_bad, int k, int n,
                                     float min_deviation) {

    __shared__ int min_value;
    min_value = 1000000.;

    __syncthreads();

    if (d_termination_criterion[0] == 0) {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            atomicMin(&min_value, d_C_sizes_best[i]);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            if (d_C_sizes_best[i] == min_value) {
                d_bad[i] = true;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            if (d_C_sizes_best[i] < n / k * min_deviation) {
                d_bad[i] = true;
            }
        }
    }
}

void
gpu_update_best(float *d_cost, float *d_cost_best,
                int *d_termination_criterion,
                int *d_M_best, int *d_M_current,
                int *d_C, int *d_C_sizes, int *d_C_best, int *d_C_sizes_best,
                bool *d_bad,
                float min_deviation, int n, int k) {

    gpu_update_best_kernel_is_best << < 1, 1 >> > (d_cost, d_cost_best, d_termination_criterion);
    gpu_update_best_kernel_init_k << < 1, k >> > (d_termination_criterion, d_M_best, d_M_current, d_bad, k);
    gpu_update_best_kernel_C << < k, BLOCK_SIZE >> >
                                     (d_C_best, d_C_sizes_best, d_C, d_C_sizes, d_termination_criterion, n);
    gpu_update_best_kernel_find_bad << < 1, k >> >
                                            (d_C_sizes_best, d_termination_criterion, d_bad, k, n, min_deviation);

}

__global__
void gpu_replace_medoids_kernel(int *d_M_current, int *d_M_random, int *d_M, int *d_M_best, bool *d_bad,
                                int k) {

    int j = 0;
    for (int i = 0; i < k; i++) {
        if (!d_bad[i]) {
            d_M_current[j] = d_M_best[i];
            j += 1;
        }
    }

    int p = 0;
    while (j < k) {
        bool is_in = false;
        for (int i = 0; i < j; i++) {
            if (d_M[d_M_random[p]] == d_M_current[i]) {
                is_in = true;
                break;
            }
        }
        if (!is_in) {
            d_M_current[j] = d_M[d_M_random[p]];
            j += 1;
        }
        p += 1;
    }
}

__global__
void remove_outliers_kernel_min_delta(float *d_delta, bool *d_D, int *d_M_best, float *d_data, int d, int k) {
    for (int i = blockIdx.x; i < k; i += gridDim.x) {
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            if (i != j) {
                float msd = 0.;
                int size = 0;
                for (int l = 0; l < d; l++) {//todo could be parallelized
                    if (d_D[i * d + l]) {
                        msd += std::abs(d_data[d_M_best[i] * d + l] - d_data[d_M_best[j] * d + l]);
                        size++;
                    }
                }
                msd /= size;

                atomicMin(&d_delta[i], msd);
            }
        }
    }
}

__global__
void
remove_outliers_kernel_remove(int *d_C_result, int *d_C_best, int *d_C_sizes_best,
                              float *d_delta,
                              bool *d_D,
                              int *d_M_best,
                              float *d_data,
                              int n, int d, int k) {

    int i = blockIdx.x;

    int C_i_size = d_C_sizes_best[i];
    for (int p_id = threadIdx.x; p_id < C_i_size; p_id += blockDim.x) {
        int p = d_C_best[i * n + p_id];

        int clustered = -1;
        for (int l = 0; l < k; l++) {

            float msd = 0.;
            int size = 0;
            for (int j = 0; j < d; j++) {
                if (d_D[l * d + j]) {
                    msd += std::abs(d_data[d_M_best[l] * d + j] - d_data[p * d + j]);
                    size++;
                }
            }
            msd /= size;

            if (msd <= d_delta[l]) {
                clustered = i;
                break;
            }
        }

        d_C_result[p] = clustered;

    }
}

void remove_outliers(int *d_C_result, int *d_C_best, int *d_C_sizes_best,
                     bool *d_D,
                     float *d_delta,
                     int *d_M_best,
                     float *d_data,
                     int n, int d, int k) {

    set_all << < 1, k >> > (d_delta, 1000000., k);//todo not nice

    remove_outliers_kernel_min_delta << < k, min(k, BLOCK_SIZE) >> > (d_delta,
            d_D,
            d_M_best,
            d_data,
            d, k);

    remove_outliers_kernel_remove << < k, BLOCK_SIZE >> > (d_C_result, d_C_best, d_C_sizes_best,
            d_delta,
            d_D,
            d_M_best,
            d_data,
            n, d, k);

}

__global__
void fill_with_indices_kernel(int *d_S, int n) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += blockDim.x * gridDim.x) {
        d_S[p] = p;
    }
}

void fill_with_indices(int *d_S, int n) {
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    fill_with_indices_kernel << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_S, n);
}

std::vector <at::Tensor>
GPU_PROCLUS(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug) {
    cudaDeviceSynchronize();
//    cudaProfilerStart();


    //getting constants
    int n = data.size(0);
    int d = data.size(1);
    l = min(l, d);
    int Ak = min(n, int(a * k));
    int Bk = min(n, int(b * k));

    float *d_data = copy_to_flatten_device(data, n, d);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;

    //initializing random generator for cuda
    curandState *d_state;
    cudaMalloc(&d_state, BLOCK_SIZE * sizeof(curandState));
    init_seed << < 1, BLOCK_SIZE >> > (d_state, 42);

    int *d_state_fixed;
    if (debug) {
        cudaMalloc(&d_state_fixed, BLOCK_SIZE * sizeof(int));
        fill_with_indices(d_state_fixed, BLOCK_SIZE);
    }

    //initializing cuda arrays
    bool *d_bad = device_allocate_bool(k);
    int *d_C = device_allocate_int(k * n);
    int *d_C_sizes = device_allocate_int(k);
    int *d_C_best = device_allocate_int(n * k);
    int *d_C_sizes_best = device_allocate_int(k);
    int *d_C_result = device_allocate_int(n);
    float *d_cost = device_allocate_float(1);
    float *d_cost_best = device_allocate_float(1);
    bool *d_D = device_allocate_bool(k * d);
    int *d_Ds = device_allocate_int(k * d);
    int *d_D_sizes = device_allocate_int(k);
    float *d_delta = device_allocate_float(k);
    float *d_dist_n_k = device_allocate_float_zero(n * k);
    int *d_L = device_allocate_int(n * k);
    int *d_L_sizes = device_allocate_int(k);
    int *d_lambda = device_allocate_int(k);
    int *d_lock = device_allocate_int(n);
    int *d_M_best = device_allocate_int(k);
    int *d_M_current = device_allocate_int(k);
    int *d_M_random = device_allocate_int(Bk);
    int *d_S = device_allocate_int(n);
    float *d_sigma = device_allocate_float(k);
    int *d_termination_criterion = device_allocate_int_zero(1);
    float *d_X = device_allocate_float(k * d);
    float *d_Z = device_allocate_float(k * d);

    //// Initialization Phase ////
    fill_with_indices(d_S, n);

    if (debug) {
        gpu_not_random_sample_locked(d_S, Ak, n, d_state_fixed, d_lock);
    } else {
        gpu_random_sample_locked(d_S, Ak, n, d_state, d_lock);
    }

    if (debug) {
        printf("d_S:\n");
        print_array_gpu(d_S, Ak);
    }

    int *d_M = gpu_greedy(d_data, d_S, Bk, Ak, d, n);

    if (debug) {
        printf("d_M:\n");
        print_array_gpu(d_M, Bk);
    }

    //// Iterative Phase ////
    fill_with_indices(d_M_random, Bk);
    if (debug) {
        gpu_not_random_sample_locked(d_M_random, k, Bk, d_state_fixed, d_lock);
    } else {
        gpu_random_sample_locked(d_M_random, k, Bk, d_state, d_lock);
    }
    gpu_gather_1d(d_M_current, d_M, d_M_random, k);
    cudaMemcpy(d_M_best, d_M_current, k * sizeof(int), cudaMemcpyDeviceToDevice);

    int termination_criterion = 0;
    set(d_cost_best, 0, 1000000.);

    while (termination_criterion < termination_rounds) {

        if (debug) {
            printf("\n\n--------------\n");
        }

        //// compute L ////
        gpu_compute_L(d_L, d_L_sizes,
                      d_dist_n_k,
                      d_delta,
                      d_M_current,
                      d_data,
                      n, d, k);

        //// find dimensions ////
        gpu_find_dimensions(d_D, d_Z, d_X,
                            d_L, d_L_sizes,
                            d_M_current,
                            d_data,
                            n, d, k, l);

        //// assign points /////
        gpu_assign_points(d_C, d_C_sizes,
                          d_D, d_Ds, d_D_sizes,
                          d_M_current,
                          d_data,
                          n, d, k);

        //// evaluate clustering ////
        gpu_evaluate_cluster(d_cost,
                             d_C, d_C_sizes,
                             d_D, d_D_sizes,
                             d_data,
                             n, d, k);


        if (debug) {
//            printf("d_delta: ");
//            print_array_gpu(d_delta, k);
            printf("d_C_sizes: ");
            print_array_gpu(d_C_sizes, k);
//            printf("d_termination_criterion: ");
//            print_array_gpu(d_termination_criterion, 1);
            printf("d_D: ");
            print_array_gpu(d_D, k, d);
            printf("d_cost: ");
            print_array_gpu(d_cost, 1);
            printf("d_M_current: ");
            print_array_gpu(d_M_current, k);
        }

        //// update best ////
        termination_criterion += 1;

        gpu_update_best(d_cost, d_cost_best,
                        d_termination_criterion,
                        d_M_best, d_M_current,
                        d_C, d_C_sizes, d_C_best, d_C_sizes_best,
                        d_bad,
                        min_deviation, n, k);

        if (debug) {
            printf("d_bad: ");
            print_array_gpu(d_bad, k);
            printf("\n--------------\n\n");
        }

        if (termination_criterion >= termination_rounds) {
            //only read from device version of termination_criterion as few times as possible
            cudaMemcpy(&termination_criterion, d_termination_criterion, sizeof(int), cudaMemcpyDeviceToHost);
        }

        //replace bad medoids
        if (debug) {
            gpu_not_random_sample_locked(d_M_random, k, Bk, d_state_fixed, d_lock);
        } else {
            gpu_random_sample_locked(d_M_random, k, Bk, d_state, d_lock);
        }
        gpu_replace_medoids_kernel << < 1, 1 >> > (d_M_current, d_M_random, d_M, d_M_best, d_bad, k);

    }

    //// Refinement Phase ////
    gpu_find_dimensions(d_D, d_Z, d_X,
                        d_C_best, d_C_sizes_best,
                        d_M_best,
                        d_data,
                        n, d, k, l);

    gpu_assign_points(d_C_best, d_C_sizes_best,
                      d_D, d_Ds, d_D_sizes,
                      d_M_best,
                      d_data,
                      n, d, k);

    remove_outliers(d_C_result, d_C_best, d_C_sizes_best,
                    d_D,
                    d_delta,
                    d_M_best,
                    d_data,
                    n, d, k);

    // building result
    std::vector <at::Tensor> r;

    torch::Tensor M_Tensor = torch::zeros({k}, torch::kInt32);
    torch::Tensor D_Tensor = torch::zeros({k, d}, torch::kBool);
    torch::Tensor C_Tensor = torch::zeros({n}, torch::kInt32);

    cudaMemcpy(M_Tensor.data_ptr<int>(), d_M_best, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(D_Tensor.data_ptr<bool>(), d_D, k * d * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_Tensor.data_ptr<int>(), d_C_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    r.push_back(M_Tensor);
    r.push_back(D_Tensor);
    r.push_back(C_Tensor);

    // free all
    cudaFree(d_bad);
    cudaFree(d_C);
    cudaFree(d_C_sizes);
    cudaFree(d_C_best);
    cudaFree(d_C_sizes_best);
    cudaFree(d_C_result);
    cudaFree(d_cost);
    cudaFree(d_cost_best);
    cudaFree(d_D);
    cudaFree(d_Ds);
    cudaFree(d_D_sizes);
    cudaFree(d_data);
    cudaFree(d_delta);
    cudaFree(d_dist_n_k);
    cudaFree(d_L);
    cudaFree(d_L_sizes);
    cudaFree(d_lambda);
    cudaFree(d_lock);
    cudaFree(d_M);
    cudaFree(d_M_best);
    cudaFree(d_M_current);
    cudaFree(d_M_random);
    cudaFree(d_S);
    cudaFree(d_sigma);
    cudaFree(d_state);
    cudaFree(d_termination_criterion);
    cudaFree(d_X);
    cudaFree(d_Z);

//    cudaProfilerStop();

    if (debug) {
        cudaFree(d_state_fixed);
    }

    cudaDeviceSynchronize();

    return r;
}

__global__
void gpu_compute_L_kernel_KEEP_dist(float *d_dist_n_k,
                                    int *d_M_current, int *d_M_bad,
                                    float *d_data,
                                    int n, int d) {
    int i = d_M_bad[blockIdx.x];
    int m_i = d_M_current[i];

    extern __shared__ float s_data_i[];

    if (threadIdx.x < d) {
        s_data_i[threadIdx.x] = d_data[m_i * d + threadIdx.x];
    }

    __syncthreads();

    for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < n; p += gridDim.y * blockDim.x) {//independent
        float sum = 0;
        for (int j = 0; j < d; j++) {//we have plenty to parallelize over here - so we can avoid the atomic
            float sub = d_data[p * d + j] - s_data_i[j];
            sum += sub * sub;
        }
        d_dist_n_k[i * n + p] = std::sqrt(sum);
    }
}

__global__
void gpu_compute_L_kernel_KEEP_L(int *d_lambda, int *d_L, int *d_L_sizes, int *d_L_sizes_change,
                                 float *d_dist_n_k,
                                 float *d_delta, float *d_delta_old,
                                 int n) {
    int i = blockIdx.x;

    if (d_delta_old[i] > d_delta[i]) {
        if (threadIdx.x == 0)
            d_lambda[i] = -1;
        for (int p = threadIdx.x; p < n; p += blockDim.x) {
            if (d_dist_n_k[i * n + p] <= d_delta_old[i] &&
                d_dist_n_k[i * n + p] > d_delta[i]) {
                int idx = atomicInc((unsigned int *) &d_L_sizes_change[i], n);
                d_L[i * n + idx] = p;
            }
        }
    } else {
        if (threadIdx.x == 0)
            d_lambda[i] = 1;
        for (int p = threadIdx.x; p < n; p += blockDim.x) {
            if (d_dist_n_k[i * n + p] > d_delta_old[i] &&
                d_dist_n_k[i * n + p] <= d_delta[i]) {
                int idx = atomicInc((unsigned int *) &d_L_sizes_change[i], n);
                d_L[i * n + idx] = p;
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        d_L_sizes[i] += d_lambda[i] * d_L_sizes_change[i];
    }
}

void gpu_compute_L_keep(int *d_L, int *d_L_sizes_change, int *d_L_sizes, int *d_lambda,
                        float *d_dist_n_k,
                        float *d_delta_old, float *d_delta,
                        int *d_M_current, int *d_M_bad, int num_bad,
                        float *d_data,
                        int n, int d, int k) {
    int number_of_blocks = n / BLOCK_SIZE_SMALL;
    if (n % BLOCK_SIZE_SMALL) number_of_blocks++;
    dim3 grid_k_n(num_bad, number_of_blocks);
    gpu_compute_L_kernel_KEEP_dist << < grid_k_n, min(n, BLOCK_SIZE_SMALL), d * sizeof(float) >> > (d_dist_n_k,
            d_M_current, d_M_bad,
            d_data,
            n, d);

    //compute delta
    gpu_compute_L_kernel_compute_delta_V2 << < 1, k >> > (d_delta, d_dist_n_k, d_M_current, n, k);

    //compute L
    cudaMemset(d_L_sizes_change, 0, k * sizeof(int));
    gpu_compute_L_kernel_KEEP_L << < k, min(n, BLOCK_SIZE) >> > (d_lambda, d_L, d_L_sizes, d_L_sizes_change,
            d_dist_n_k,
            d_delta, d_delta_old,
            n);

    cudaMemcpy(d_delta_old, d_delta, k * sizeof(float), cudaMemcpyDeviceToDevice);
}


__global__
void
gpu_find_dimensions_kernel_KEEP_H(float *d_H,
                                  int *d_L, int *d_L_sizes_change, int *d_lambda,
                                  int *d_M_current,
                                  float *d_data,
                                  int n, int d) {
    int i = blockIdx.x; //independent for different k
    int j = threadIdx.x; //independent for different d

    float sum = 0.;

    int m_i = d_M_current[i];
    int L_i_size_change = d_L_sizes_change[i];
    float data_ij = d_data[m_i * d + j];

    for (int p = blockDim.y * blockIdx.y + threadIdx.y; p < L_i_size_change; p += gridDim.y * blockDim.y) {
        int point = d_L[i * n + p];
        sum += std::abs(d_data[point * d + j] - data_ij);
    }

    atomicAdd(&d_H[i * d + j], d_lambda[i] * sum);
}


__global__
void
gpu_find_dimensions_kernel_KEEP_X(float *d_X, float *d_H, int *d_L_sizes, int d) {

    int i = blockIdx.x; //independent for different k
    int j = threadIdx.x; //independent for different d
    int L_i_size = d_L_sizes[i];

    d_X[i * d + j] = d_H[i * d + j] / L_i_size;
}

void gpu_find_dimensions_keep(bool *d_D, float *d_Z, float *d_X, float *d_H,
                              int *d_L, int *d_L_sizes_change, int *d_L_sizes, int *d_lambda,
                              int *d_M_current,
                              float *d_data,
                              int n, int d, int k, int l) {
    int number_of_blocks = (k * d) / BLOCK_SIZE;
    if ((k * d) % BLOCK_SIZE) number_of_blocks++;

    set_all << < number_of_blocks, min(k * d, BLOCK_SIZE) >> > (d_X, 0, k * d);

    int remaining_d = BLOCK_SIZE / d;
    int number_of_blocks_X_join_v2 = (n / k) / remaining_d;
    if ((n / k) % remaining_d) number_of_blocks_X_join_v2++;
    dim3 grid_X_join_v2(k, number_of_blocks_X_join_v2);
    dim3 block_X_join_v2(d, remaining_d);


    gpu_find_dimensions_kernel_KEEP_H << < grid_X_join_v2, block_X_join_v2 >> > (d_H,
            d_L, d_L_sizes_change, d_lambda,
            d_M_current,
            d_data,
            n, d);

    gpu_find_dimensions_kernel_KEEP_X << < k, d >> > (d_X, d_H, d_L_sizes, d);

    gpu_find_dimensions_kernel_Z << < k, d >> > (d_Z, d_X, k, d);


    //compute D
    set_all << < number_of_blocks, min(k * d, BLOCK_SIZE) >> > (d_D, false, k * d);
    dim3 block(min(32, k), min(32, d));
    gpu_find_dimensions_kernel_compute_D << < 1, block, 2 * k * sizeof(float) >> > (d_D, d_Z, k, d, l);
}

__global__
void gpu_replace_medoids_kernel_Keep(int *d_M_bad, int *d_num_bad, int *d_M_current, int *d_M_random, int *d_M,
                                     int *d_M_best,
                                     bool *d_bad, int k) {

    extern __shared__ int s_M_kept[];

    int j = 0;
    for (int i = 0; i < k; i++) {
        if (!d_bad[i]) {
            d_M_current[i] = d_M_best[i];
            s_M_kept[j] = d_M_best[i];
            j += 1;
        }
    }

    int p = 0;
    int l = 0;
    for (int i = 0; i < k; i++) {
        if (d_bad[i]) {
            d_M_bad[l] = i;//todo explain why this is needed in the GPU version
            l++;

            bool is_in = true;
            while (is_in) {
                is_in = false;
                for (int q = 0; q < j; q++) {
                    if (d_M[d_M_random[p]] == s_M_kept[q]) {
                        is_in = true;
                        p++;
                        break;
                    }
                }
            }
            d_M_current[i] = d_M[d_M_random[p]];
            s_M_kept[j] = d_M[d_M_random[p]];
            j++;
            p++;
        }
    }

    d_num_bad[0] = l;
}

__global__
void gpu_replace_medoids_kernel_keep_reset(int *d_L_sizes, float *d_delta_old, float *d_H, int *d_M_bad, int d) {
    int i = d_M_bad[blockIdx.x];
    int j = threadIdx.x;

    d_L_sizes[i] = 0;
    d_delta_old[i] = -1.;
    d_H[i * d + j] = 0.;
}

std::vector <at::Tensor>
GPU_PROCLUS_KEEP(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds,
                 bool debug) {
    cudaDeviceSynchronize();
//    cudaProfilerStart();

    //getting constants
    int n = data.size(0);
    int d = data.size(1);
    l = min(l, d);
    int Ak = min(n, int(a * k));
    int Bk = min(n, int(b * k));

    //copying data to the GPU
    float *d_data = copy_to_flatten_device(data, n, d);

    //initializing random generator for cuda
    curandState *d_state;
    cudaMalloc(&d_state, BLOCK_SIZE * sizeof(curandState));
    init_seed << < 1, BLOCK_SIZE >> > (d_state, 42);

    int *d_state_fixed;
    if (debug) {
        cudaMalloc(&d_state_fixed, BLOCK_SIZE * sizeof(int));
        fill_with_indices(d_state_fixed, BLOCK_SIZE);
    }

    //initializing cuda arrays
    bool *d_bad = device_allocate_bool(k);
    int *d_C = device_allocate_int(k * n);
    int *d_C_sizes = device_allocate_int(k);
    int *d_C_best = device_allocate_int(n * k);
    int *d_C_sizes_best = device_allocate_int(k);
    int *d_C_result = device_allocate_int(n);
    float *d_cost = device_allocate_float(1);
    float *d_cost_best = device_allocate_float(1);
    bool *d_D = device_allocate_bool(k * d);
    int *d_Ds = device_allocate_int(k * d);
    int *d_D_sizes = device_allocate_int(k);
    float *d_delta = device_allocate_float(k);
    float *d_delta_old = device_allocate_float(k);
    float *d_dist_n_k = device_allocate_float_zero(n * k);
    float *d_H = device_allocate_float_zero(k * d);
    int *d_L = device_allocate_int(n * k);
    int *d_L_sizes = device_allocate_int_zero(k);
    int *d_L_sizes_change = device_allocate_int(k);
    int *d_lambda = device_allocate_int(k);
    int *d_lock = device_allocate_int(n);
    int *d_M_best = device_allocate_int(k);
    int *d_M_current = device_allocate_int(k);
    int *d_M_random = device_allocate_int(Bk);
    int *d_M_bad = device_allocate_int(k);
    int *d_num_bad = device_allocate_int(1);
    int num_bad = k;
    int *d_S = device_allocate_int(n);
    float *d_sigma = device_allocate_float(k);
    int *d_termination_criterion = device_allocate_int_zero(1);
    float *d_X = device_allocate_float(k * d);
    float *d_Z = device_allocate_float(k * d);

    //// Initialization Phase ////
    fill_with_indices(d_S, n);
    if (debug) {
        gpu_not_random_sample_locked(d_S, Ak, n, d_state_fixed, d_lock);
    } else {
        gpu_random_sample_locked(d_S, Ak, n, d_state, d_lock);
    }

    int *d_M = gpu_greedy(d_data, d_S, Bk, Ak, d, n);

    //// Iterative Phase ///
    fill_with_indices(d_M_random, Bk);

    if (debug) {
        gpu_not_random_sample_locked(d_M_random, k, Bk, d_state_fixed, d_lock);
    } else {
        gpu_random_sample_locked(d_M_random, k, Bk, d_state, d_lock);
    }
    gpu_gather_1d(d_M_current, d_M, d_M_random, k);
    cudaMemcpy(d_M_best, d_M_current, k * sizeof(int), cudaMemcpyDeviceToDevice);
    fill_with_indices(d_M_bad, k);

    int termination_criterion = 0;
    set(d_cost_best, 0, 1000000.);

    int number_of_blocks = k / BLOCK_SIZE;
    if (k % BLOCK_SIZE) number_of_blocks++;
    set_all << < number_of_blocks, min(k, BLOCK_SIZE) >> > (d_delta_old, -1., k);

    while (termination_criterion < termination_rounds) {
        if (debug) {
            printf("\n\n--------------\n");
        }

        //// compute L ////
        gpu_compute_L_keep(d_L, d_L_sizes_change, d_L_sizes, d_lambda,
                           d_dist_n_k,
                           d_delta_old, d_delta,
                           d_M_current, d_M_bad, num_bad,
                           d_data,
                           n, d, k);

        //// find dimensions ////
        gpu_find_dimensions_keep(d_D, d_Z, d_X, d_H,
                                 d_L, d_L_sizes_change, d_L_sizes, d_lambda,
                                 d_M_current,
                                 d_data,
                                 n, d, k, l);

        //// assign points /////
        gpu_assign_points(d_C, d_C_sizes,
                          d_D, d_Ds, d_D_sizes,
                          d_M_current,
                          d_data,
                          n, d, k);

        //// evaluate clustering ////
        gpu_evaluate_cluster(d_cost,
                             d_C, d_C_sizes,
                             d_D, d_D_sizes,
                             d_data,
                             n, d, k);


        if (debug) {
            printf("d_C_sizes: ");
            print_array_gpu(d_C_sizes, k);
            printf("d_termination_criterion: ");
            print_array_gpu(d_termination_criterion, 1);
            printf("d_D: ");
            print_array_gpu(d_D, k, d);
            printf("d_cost: ");
            print_array_gpu(d_cost, 1);
            printf("d_M_current: ");
            print_array_gpu(d_M_current, k);
            printf("\n--------------\n\n");
        }

        //// update best ////
        termination_criterion += 1;
        gpu_update_best(d_cost, d_cost_best,
                        d_termination_criterion,
                        d_M_best, d_M_current,
                        d_C, d_C_sizes, d_C_best, d_C_sizes_best,
                        d_bad,
                        min_deviation, n, k);

        if (debug) {
            printf("d_bad: ");
            print_array_gpu(d_bad, k);
        }

        if (termination_criterion >= termination_rounds) {
            //only read from device version of termination_criterion as few times as possible
            cudaMemcpy(&termination_criterion, d_termination_criterion, sizeof(int), cudaMemcpyDeviceToHost);
        }

        //replace bad medoids
        if (debug) {
            gpu_not_random_sample_locked(d_M_random, k, Bk, d_state_fixed, d_lock);
        } else {
            gpu_random_sample_locked(d_M_random, k, Bk, d_state, d_lock);
        }

        gpu_replace_medoids_kernel_Keep << < 1, 1, k * sizeof(int) >> >
                                                   (d_M_bad, d_num_bad, d_M_current, d_M_random, d_M,
                                                           d_M_best, d_bad, k);
        cudaMemcpy(&num_bad, d_num_bad, sizeof(int), cudaMemcpyDeviceToHost);
        gpu_replace_medoids_kernel_keep_reset << < num_bad, d >> > (d_L_sizes, d_delta_old, d_H, d_M_bad, d);

    }

    //// Refinement Phase ////
    gpu_find_dimensions(d_D, d_Z, d_X,
                        d_C_best, d_C_sizes_best,
                        d_M_best,
                        d_data,
                        n, d, k, l);

    gpu_assign_points(d_C_best, d_C_sizes_best,
                      d_D, d_Ds, d_D_sizes,
                      d_M_best,
                      d_data,
                      n, d, k);

    remove_outliers(d_C_result, d_C_best, d_C_sizes_best,
                    d_D,
                    d_delta,
                    d_M_best,
                    d_data,
                    n, d, k);

    // building result
    std::vector <at::Tensor> r;

    torch::Tensor M_Tensor = torch::zeros({k}, torch::kInt32);
    torch::Tensor D_Tensor = torch::zeros({k, d}, torch::kBool);
    torch::Tensor C_Tensor = torch::zeros({n}, torch::kInt32);

    cudaMemcpy(M_Tensor.data_ptr<int>(), d_M_best, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(D_Tensor.data_ptr<bool>(), d_D, k * d * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_Tensor.data_ptr<int>(), d_C_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    r.push_back(M_Tensor);
    r.push_back(D_Tensor);
    r.push_back(C_Tensor);

    // free all
    cudaFree(d_bad);
    cudaFree(d_C);
    cudaFree(d_C_sizes);
    cudaFree(d_C_best);
    cudaFree(d_C_sizes_best);
    cudaFree(d_C_result);
    cudaFree(d_cost);
    cudaFree(d_cost_best);
    cudaFree(d_D);
    cudaFree(d_Ds);
    cudaFree(d_D_sizes);
    cudaFree(d_data);
    cudaFree(d_delta);
    cudaFree(d_delta_old);
    cudaFree(d_dist_n_k);
    cudaFree(d_H);
    cudaFree(d_L);
    cudaFree(d_L_sizes);
    cudaFree(d_L_sizes_change);
    cudaFree(d_lambda);
    cudaFree(d_lock);
    cudaFree(d_M);
    cudaFree(d_M_best);
    cudaFree(d_M_current);
    cudaFree(d_M_random);
    cudaFree(d_S);
    cudaFree(d_sigma);
    cudaFree(d_state);
    cudaFree(d_termination_criterion);
    cudaFree(d_X);
    cudaFree(d_Z);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
//    cudaProfilerStop();

    return r;
}


__global__
void
gpu_compute_L_kernel_sum_dist_SAVE(const int *__restrict__ d_M_idx, const int *__restrict__ d_M,
                                   const float *__restrict__ d_data, float *__restrict__ d_dist_n_Bk,
                                   const bool *__restrict__ d_dist_n_Bk_set,
                                   const int k, const int d, const int n) {
    int l = blockIdx.x;
    int i = d_M_idx[l];
    int m_i = d_M[i];

    extern __shared__ float s_data_i[];

    if (!d_dist_n_Bk_set[i]) {
        if (threadIdx.x < d) {
            s_data_i[threadIdx.x] = d_data[m_i * d + threadIdx.x];
        }

        __syncthreads();

        for (int p = blockIdx.y * blockDim.x + threadIdx.x; p < n; p += gridDim.y * blockDim.x) {//independent
            float sum = 0;
            for (int j = 0; j < d; j++) {//we have plenty to parallelize over here - so we can avoid the atomic
                float sub = d_data[p * d + j] - s_data_i[j];
                sum += sub * sub;
            }
            d_dist_n_Bk[i * n + p] = std::sqrt(sum);
        }
    }
}

__global__
void gpu_compute_L_kernel_sqrt_dist_pre_mark(int *d_M_idx, float *d_dist_n_Bk, bool *d_dist_n_Bk_set, int k, int n) {
    for (int l = threadIdx.x; l < k; l += blockDim.x) {//independent
        int i = d_M_idx[l];
        d_dist_n_Bk_set[i] = true;
    }
}

__global__
void gpu_compute_L_kernel_compute_delta_pre(int *d_M_idx, int *d_M, float *d_delta, float *d_dist, int k, int n) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {//independent
        d_delta[i] = 1000000.;//todo not nice
        for (int p = 0; p < k; p++) {
            if (i != p) {
                if (d_dist[d_M_idx[i] * n + d_M[d_M_idx[p]]] < d_delta[i]) {//todo be carefull here
                    d_delta[i] = d_dist[d_M_idx[i] * n + d_M[d_M_idx[p]]];
                }
            }
        }
    }
}

__global__
void
gpu_compute_L_kernel_compute_L_pre(int *d_posneg, int *d_M_idx, int *d_L, int *d_L_sizes, int *d_L_sizes_change,
                                   float *d_dist_n_Bk, float *d_delta, float *d_old_delta, int k, int n) {

    for (int i = blockIdx.x; i < k; i += gridDim.x) {//independent
        if (d_old_delta[d_M_idx[i]] > d_delta[i]) {
            if (threadIdx.x == 0)
                d_posneg[i] = -1;
            for (int p = threadIdx.x; p < n; p += blockDim.x) {
                if (d_dist_n_Bk[d_M_idx[i] * n + p] <= d_old_delta[d_M_idx[i]] &&
                    d_dist_n_Bk[d_M_idx[i] * n + p] > d_delta[i]) {
                    int old_size = atomicInc((unsigned int *) &d_L_sizes_change[i], n);
                    d_L[i * n + old_size] = p;
                }
            }
        } else {
            if (threadIdx.x == 0)
                d_posneg[i] = 1;
            for (int p = threadIdx.x; p < n; p += blockDim.x) {
                if (d_dist_n_Bk[d_M_idx[i] * n + p] > d_old_delta[d_M_idx[i]] &&
                    d_dist_n_Bk[d_M_idx[i] * n + p] <= d_delta[i]) {
                    int old_size = atomicInc((unsigned int *) &d_L_sizes_change[i], n);
                    d_L[i * n + old_size] = p;
                }
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            d_L_sizes[d_M_idx[i]] += d_posneg[i] * d_L_sizes_change[i];
        }
    }
}

__global__
void gpu_compute_L_kernel_set_old_delta_pre(int *d_M_idx, float *d_old_delta, float *d_delta, int k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        d_old_delta[d_M_idx[i]] = d_delta[i];
    }
}

void gpu_compute_L_save(int *d_L, int *d_L_sizes_change, int *d_L_sizes, int *d_lambda,
                        float *d_dist_n_Bk, bool *d_dist_n_Bk_set,
                        float *d_delta_old, float *d_delta,
                        int *d_M, int *d_M_idx,
                        float *d_data,
                        int n, int d, int k) {
    int number_of_blocks = n / BLOCK_SIZE_SMALL;
    if (n % BLOCK_SIZE) number_of_blocks++;
    dim3 grid_k_n(k, number_of_blocks);
    gpu_compute_L_kernel_sum_dist_SAVE << < grid_k_n, min(n, BLOCK_SIZE_SMALL), d * sizeof(float) >> > (d_M_idx, d_M,
            d_data,
            d_dist_n_Bk,
            d_dist_n_Bk_set,
            k, d, n);

    gpu_compute_L_kernel_sqrt_dist_pre_mark << < 1, k >> > (d_M_idx, d_dist_n_Bk, d_dist_n_Bk_set, k, n);

    //compute delta
    gpu_compute_L_kernel_compute_delta_pre << < 1, k >> > (d_M_idx, d_M, d_delta, d_dist_n_Bk, k, n);

    //compute L
    cudaMemset(d_L_sizes_change, 0, k * sizeof(int));
    gpu_compute_L_kernel_compute_L_pre << < k, min(n, BLOCK_SIZE) >> > (d_lambda, d_M_idx, d_L, d_L_sizes,
            d_L_sizes_change,
            d_dist_n_Bk, d_delta, d_delta_old, k, n);

    gpu_compute_L_kernel_set_old_delta_pre << < 1, k >> > (d_M_idx, d_delta_old, d_delta, k);
}


__global__
void
gpu_find_dimensions_kernel_SAVE_H(float *__restrict__ d_X, float *__restrict__ d_H,
                                  const float *__restrict__ d_data,
                                  const int *__restrict__ d_L,
                                  const int *__restrict__ d_L_sizes_change,
                                  const int *__restrict__ d_L_sizes,
                                  const int *__restrict__ d_lambda,
                                  const int *__restrict__ d_M_current,
                                  const int *__restrict__ d_M_idx,
                                  const int k, const int d, const int n) {
    int i = blockIdx.x; //independent for different k
    int j = threadIdx.x; //independent for different d

    float sum = 0.;

    int m_i = d_M_current[i];
    int m_idx = d_M_idx[i];
    int L_i_size_change = d_L_sizes_change[i];
    int posneg_i = d_lambda[i];
    float data_ij = d_data[m_i * d + j];

    for (int p = blockDim.y * blockIdx.y + threadIdx.y; p < L_i_size_change; p += gridDim.y * blockDim.y) {
        int point = d_L[i * n + p];
        sum += std::abs(d_data[point * d + j] - data_ij);
    }

    atomicAdd(&d_H[m_idx * d + j], posneg_i * sum);
}


__global__
void
gpu_find_dimensions_kernel_SAVE_X(float *__restrict__ d_X, float *__restrict__ d_H,
                                  const float *__restrict__ d_data,
                                  const int *__restrict__ d_L,
                                  const int *__restrict__ d_L_sizes_change,
                                  const int *__restrict__ d_L_sizes,
                                  const int *__restrict__ d_lambda,
                                  const int *__restrict__ d_M_current,
                                  const int *__restrict__ d_M_idx,
                                  const int k, const int d, const int n) {

    int i = blockIdx.x; //independent for different k
    int j = threadIdx.x; //independent for different d
    int m_idx = d_M_idx[i];
    int L_i_size = d_L_sizes[m_idx];

    d_X[i * d + j] = d_H[m_idx * d + j] / L_i_size;
}

void gpu_find_dimensions_save(bool *d_D, float *d_Z, float *d_X, float *d_H,
                              int *d_L, int *d_L_sizes_change, int *d_L_sizes, int *d_lambda,
                              int *d_M_current, int *d_M_idx,
                              float *d_data,
                              int n, int d, int k, int l) {

    int number_of_blocks = (k * d) / BLOCK_SIZE;
    if ((k * d) % BLOCK_SIZE) number_of_blocks++;
    set_all << < number_of_blocks, min(k * d, BLOCK_SIZE) >> > (d_X, 0, k * d);


    int remaining_d = BLOCK_SIZE / d;
    int number_of_blocks_X_join_v2 = (n / k) / remaining_d;
    if ((n / k) % remaining_d) number_of_blocks_X_join_v2++;
    dim3 grid_X_join_v2(k, number_of_blocks_X_join_v2);
    dim3 block_X_join_v2(d, remaining_d);


    gpu_find_dimensions_kernel_SAVE_H << < grid_X_join_v2, block_X_join_v2 >> > (d_X, d_H, d_data, d_L,
            d_L_sizes_change, d_L_sizes, d_lambda,
            d_M_current, d_M_idx,
            k, d, n);

    gpu_find_dimensions_kernel_SAVE_X << < k, d >> > (d_X, d_H, d_data, d_L,
            d_L_sizes_change, d_L_sizes, d_lambda,
            d_M_current, d_M_idx,
            k, d, n);

    gpu_find_dimensions_kernel_Z << < k, d >> > (d_Z, d_X, k, d);

    //compute D
    set_all << < number_of_blocks, min(k * d, BLOCK_SIZE) >> > (d_D, false, k * d);
    dim3 block(min(32, k), min(32, d));
    gpu_find_dimensions_kernel_compute_D << < 1, block, 2 * k * sizeof(float) >> > (d_D, d_Z, k, d, l);
}

__global__
void
gpu_update_best_kernel_init_k_pre(int *d_M_idx, int *d_M_idx_best, int *d_termination_criterion,
                                  int *d_M_best, int *d_M_current,
                                  bool *d_bad, int k) {

    if (d_termination_criterion[0] == 0) {//todo worng!!!! then we allways pick the last????
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            d_M_best[i] = d_M_current[i];
            d_M_idx_best[i] = d_M_idx[i];
            d_bad[i] = false;
        }
    }
}

void
gpu_update_best_SAVE(int *d_M_idx, int *d_M_idx_best, float *d_cost,
                     float *d_cost_best,
                     int *d_termination_criterion,
                     int *d_M_best, int *d_M_current,
                     int *d_C, int *d_C_sizes, int *d_C_best, int *d_C_sizes_best,
                     bool *d_bad,
                     float min_deviation, int n, int k) {

    gpu_update_best_kernel_is_best << < 1, 1 >> > (d_cost, d_cost_best, d_termination_criterion);
    gpu_update_best_kernel_init_k_pre << < 1, k >> > (d_M_idx, d_M_idx_best, d_termination_criterion, d_M_best,
            d_M_current, d_bad, k);
    gpu_update_best_kernel_C << < k, BLOCK_SIZE >> >
                                     (d_C_best, d_C_sizes_best, d_C, d_C_sizes, d_termination_criterion, n);
    gpu_update_best_kernel_find_bad << < 1, k >> >
                                            (d_C_sizes_best, d_termination_criterion, d_bad, k, n, min_deviation);

}

__global__
void
gpu_replace_medoids_kernel_pre(int *d_M_idx, int *d_M_idx_best, int *d_M_current, int *d_M_random, int *d_M, int Bk,
                               int *d_M_best, bool *d_bad,
                               int k, int n) {

    int j = 0;
    for (int i = 0; i < k; i++) {
        if (!d_bad[i]) {
            d_M_current[j] = d_M_best[i];
            d_M_idx[j] = d_M_idx_best[i];
            j += 1;
        }
    }

    int p = 0;
    while (j < k) {
        bool is_in = false;
        for (int i = 0; i < j; i++) {
            if (d_M[d_M_random[p]] == d_M_current[i]) {
                is_in = true;
                break;
            }
        }
        if (!is_in) {
            d_M_current[j] = d_M[d_M_random[p]];
            d_M_idx[j] = d_M_random[p];
            j += 1;
        }
        p += 1;
    }
}

std::vector <at::Tensor>
GPU_PROCLUS_SAVE(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds,
                 bool debug) {
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
//    cudaProfilerStart();

    //getting constants
    int n = data.size(0);
    int d = data.size(1);
    l = min(l, d);
    int Ak = min(n, int(a * k));
    int Bk = min(n, int(b * k));

    //copying data to the GPU
    float *d_data = copy_to_flatten_device(data, n, d);
    gpuErrchk(cudaPeekAtLastError());

    //initializing random generator for cuda
    curandState *d_state;
    cudaMalloc(&d_state, BLOCK_SIZE * sizeof(curandState));
    init_seed << < 1, BLOCK_SIZE >> > (d_state, 42);
    gpuErrchk(cudaPeekAtLastError());

    int *d_state_fixed;
    if (debug) {
        cudaMalloc(&d_state_fixed, BLOCK_SIZE * sizeof(int));
        fill_with_indices(d_state_fixed, BLOCK_SIZE);
    }
    gpuErrchk(cudaPeekAtLastError());

    //initializing cuda arrays
    bool *d_bad = device_allocate_bool(k);
    int *d_C = device_allocate_int(k * n);
    int *d_C_sizes = device_allocate_int(k);
    gpuErrchk(cudaPeekAtLastError());
    int *d_C_best = device_allocate_int(n * k);
    gpuErrchk(cudaPeekAtLastError());
    int *d_C_sizes_best = device_allocate_int(k);
    int *d_C_result = device_allocate_int(n);
    float *d_cost = device_allocate_float(1);
    float *d_cost_best = device_allocate_float(1);
    bool *d_D = device_allocate_bool(k * d);
    int *d_Ds = device_allocate_int(k * d);
    int *d_D_sizes = device_allocate_int(k);
    float *d_delta = device_allocate_float(k);
    float *d_delta_old = device_allocate_float(Bk);
    gpuErrchk(cudaPeekAtLastError());
    float *d_dist_n_Bk = device_allocate_float_zero(n * Bk);
    gpuErrchk(cudaPeekAtLastError());
    bool *d_dist_n_Bk_set = device_allocate_bool_zero(Bk);
    float *d_H = device_allocate_float_zero(Bk * d);
    int *d_L = device_allocate_int(n * k);
    int *d_L_sizes = device_allocate_int_zero(Bk);
    int *d_L_sizes_change = device_allocate_int(k);
    int *d_lambda = device_allocate_int(k);
    int *d_lock = device_allocate_int(n);
    int *d_M_best = device_allocate_int(k);
    int *d_M_current = device_allocate_int(k);
    int *d_M_idx = device_allocate_int(k);
    int *d_M_idx_best = device_allocate_int(k);
    int *d_M_random = device_allocate_int(Bk);
    int *d_S = device_allocate_int(n);
    float *d_sigma = device_allocate_float(k);
    int *d_termination_criterion = device_allocate_int_zero(1);
    float *d_X = device_allocate_float(k * d);
    float *d_Z = device_allocate_float(k * d);
    gpuErrchk(cudaPeekAtLastError());

    //// Initialization Phase ////
    fill_with_indices(d_S, n);
    gpuErrchk(cudaPeekAtLastError());

    if (debug) {
        gpu_not_random_sample_locked(d_S, Ak, n, d_state_fixed, d_lock);
    } else {
        gpu_random_sample_locked(d_S, Ak, n, d_state, d_lock);
    }

    int *d_M = gpu_greedy(d_data, d_S, Bk, Ak, d, n);
    gpuErrchk(cudaPeekAtLastError());

    //// Iterative Phase ///
    fill_with_indices(d_M_random, Bk);
    if (debug) {
        gpu_not_random_sample_locked(d_M_random, k, Bk, d_state_fixed, d_lock);
    } else {
        gpu_random_sample_locked(d_M_random, k, Bk, d_state, d_lock);
    }
    gpu_gather_1d(d_M_current, d_M, d_M_random, k);
    cudaMemcpy(d_M_best, d_M_current, k * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_M_idx, d_M_random, k * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_M_idx_best, d_M_random, k * sizeof(int), cudaMemcpyDeviceToDevice);
    gpuErrchk(cudaPeekAtLastError());

    int termination_criterion = 0;
    set(d_cost_best, 0, 1000000.);

    int number_of_blocks = Bk / BLOCK_SIZE;
    if (Bk % BLOCK_SIZE) number_of_blocks++;
    set_all << < number_of_blocks, min(Bk, BLOCK_SIZE) >> > (d_delta_old, -1., Bk);
    gpuErrchk(cudaPeekAtLastError());

    while (termination_criterion < termination_rounds) {

        //// compute L ////
        gpu_compute_L_save(d_L, d_L_sizes_change, d_L_sizes, d_lambda,
                           d_dist_n_Bk, d_dist_n_Bk_set,
                           d_delta_old, d_delta,
                           d_M, d_M_idx,
                           d_data,
                           n, d, k);
        gpuErrchk(cudaPeekAtLastError());

        //// find dimensions ////
        gpu_find_dimensions_save(d_D, d_Z, d_X, d_H,
                                 d_L, d_L_sizes_change, d_L_sizes, d_lambda,
                                 d_M_current, d_M_idx,
                                 d_data,
                                 n, d, k, l);
        gpuErrchk(cudaPeekAtLastError());

        //// assign points /////
        gpu_assign_points(d_C, d_C_sizes,
                          d_D, d_Ds, d_D_sizes,
                          d_M_current,
                          d_data,
                          n, d, k);
        gpuErrchk(cudaPeekAtLastError());

        //// evaluate clustering ////
        gpu_evaluate_cluster(d_cost,
                             d_C, d_C_sizes,
                             d_D, d_D_sizes,
                             d_data,
                             n, d, k);
        gpuErrchk(cudaPeekAtLastError());

        if (debug) {
            printf("d_C_sizes: ");
            print_array_gpu(d_C_sizes, k);
            printf("d_termination_criterion: ");
            print_array_gpu(d_termination_criterion, 1);
            printf("d_D: ");
            print_array_gpu(d_D, k, d);
            printf("d_cost: ");
            print_array_gpu(d_cost, 1);
            printf("d_M_current: ");
            print_array_gpu(d_M_current, k);
        }

        //// update best ////
        termination_criterion += 1;
        gpu_update_best_SAVE(d_M_idx, d_M_idx_best, d_cost, d_cost_best,
                             d_termination_criterion,
                             d_M_best, d_M_current,
                             d_C, d_C_sizes, d_C_best, d_C_sizes_best,
                             d_bad,
                             min_deviation, n, k);
        gpuErrchk(cudaPeekAtLastError());

        if (debug) {
            printf("d_bad: ");
            print_array_gpu(d_bad, k);
        }

        if (termination_criterion >= termination_rounds) {
            //only read from device version of termination_criterion as few times as possible
            cudaMemcpy(&termination_criterion, d_termination_criterion, sizeof(int), cudaMemcpyDeviceToHost);
        }

        //replace bad medoids
        if (debug) {
            gpu_not_random_sample_locked(d_M_random, k, Bk, d_state_fixed, d_lock);
        } else {
            gpu_random_sample_locked(d_M_random, k, Bk, d_state, d_lock);
        }
        gpu_replace_medoids_kernel_pre << < 1, 1 >> > (d_M_idx, d_M_idx_best, d_M_current, d_M_random, d_M, Bk,
                d_M_best, d_bad, k, n);
        gpuErrchk(cudaPeekAtLastError());

    }

    //// Refinement Phase ////
    gpu_find_dimensions(d_D, d_Z, d_X,
                        d_C_best, d_C_sizes_best,
                        d_M_best,
                        d_data,
                        n, d, k, l);
    gpuErrchk(cudaPeekAtLastError());

    gpu_assign_points(d_C_best, d_C_sizes_best,
                      d_D, d_Ds, d_D_sizes,
                      d_M_best,
                      d_data,
                      n, d, k);
    gpuErrchk(cudaPeekAtLastError());

    remove_outliers(d_C_result, d_C_best, d_C_sizes_best,
                    d_D,
                    d_delta,
                    d_M_best,
                    d_data,
                    n, d, k);
    gpuErrchk(cudaPeekAtLastError());

    // building result
    std::vector <at::Tensor> r;

    torch::Tensor M_Tensor = torch::zeros({k}, torch::kInt32);
    torch::Tensor D_Tensor = torch::zeros({k, d}, torch::kBool);
    torch::Tensor C_Tensor = torch::zeros({n}, torch::kInt32);

    cudaMemcpy(M_Tensor.data_ptr<int>(), d_M_best, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(D_Tensor.data_ptr<bool>(), d_D, k * d * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_Tensor.data_ptr<int>(), d_C_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    r.push_back(M_Tensor);
    r.push_back(D_Tensor);
    r.push_back(C_Tensor);

    gpuErrchk(cudaPeekAtLastError());

    // free all
    cudaFree(d_bad);
    cudaFree(d_C);
    cudaFree(d_C_sizes);
    cudaFree(d_C_best);
    cudaFree(d_C_sizes_best);
    cudaFree(d_C_result);
    cudaFree(d_cost);
    cudaFree(d_cost_best);
    cudaFree(d_D);
    cudaFree(d_Ds);
    cudaFree(d_D_sizes);
    cudaFree(d_data);
    cudaFree(d_delta);
    cudaFree(d_delta_old);
    cudaFree(d_dist_n_Bk);
    cudaFree(d_dist_n_Bk_set);
    cudaFree(d_H);
    cudaFree(d_L);
    cudaFree(d_L_sizes);
    cudaFree(d_L_sizes_change);
    cudaFree(d_lambda);
    cudaFree(d_lock);
    cudaFree(d_M);
    cudaFree(d_M_best);
    cudaFree(d_M_current);
    cudaFree(d_M_idx);
    cudaFree(d_M_idx_best);
    cudaFree(d_M_random);
    cudaFree(d_S);
    cudaFree(d_sigma);
    cudaFree(d_state);
    cudaFree(d_termination_criterion);
    cudaFree(d_X);
    cudaFree(d_Z);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
//    cudaProfilerStop();

    return r;
}


std::vector <std::vector<at::Tensor>>
GPU_PROCLUS_PARAM(at::Tensor data, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
                  int termination_rounds) {
    cudaDeviceSynchronize();
//    cudaProfilerStart();

    //getting constants
    int k_max = ks[0];
    int l_max = ls[0];

    int n = data.size(0);
    int d = data.size(1);
    int Ak = min(n, int(a * k_max));
    int Bk = min(n, int(b * k_max));

    //copying data to the GPU
    float *d_data = copy_to_flatten_device(data, n, d);

    //initializing random generator for cuda
    curandState *d_state;
    cudaMalloc(&d_state, BLOCK_SIZE * sizeof(curandState));
    init_seed << < 1, BLOCK_SIZE >> > (d_state, 42);

    //initializing cuda arrays
    bool *d_bad = device_allocate_bool(k_max);
    int *d_C = device_allocate_int(k_max * n);
    int *d_C_sizes = device_allocate_int(k_max);
    int *d_C_best = device_allocate_int(n * k_max);
    int *d_C_sizes_best = device_allocate_int(k_max);
    int *d_C_result = device_allocate_int(n);
    float *d_cost = device_allocate_float(1);
    float *d_cost_best = device_allocate_float(1);
    bool *d_D = device_allocate_bool(k_max * d);
    int *d_Ds = device_allocate_int(k_max * d);
    int *d_D_sizes = device_allocate_int(k_max);
    float *d_delta = device_allocate_float(k_max);
    float *d_delta_old = device_allocate_float_zero(Bk);
    float *d_dist_n_Bk = device_allocate_float_zero(n * Bk);
    bool *d_dist_n_Bk_set = device_allocate_bool_zero(Bk);
    float *d_H = device_allocate_float_zero(Bk * d);
    int *d_L = device_allocate_int(n * k_max);
    int *d_L_sizes = device_allocate_int_zero(Bk);
    int *d_L_sizes_change = device_allocate_int(k_max);
    int *d_lambda = device_allocate_int(k_max);
    int *d_lock = device_allocate_int(n);
    int *d_M_best = device_allocate_int(k_max);
    int *d_M_current = device_allocate_int(k_max);
    int *d_M_idx = device_allocate_int(k_max);
    int *d_M_idx_best = device_allocate_int(k_max);
    int *d_M_random = device_allocate_int(Bk);
    int *d_S = device_allocate_int(n);
    float *d_sigma = device_allocate_float(k_max);
    int *d_termination_criterion = device_allocate_int_zero(1);
    float *d_X = device_allocate_float(k_max * d);
    float *d_Z = device_allocate_float(k_max * d);

    //// Initialization Phase ////
    fill_with_indices(d_S, n);
    gpu_random_sample_locked(d_S, Ak, n, d_state, d_lock);

    int *d_M = gpu_greedy(d_data, d_S, Bk, Ak, d, n);

    //// Iterative Phase ///
    fill_with_indices(d_M_random, Bk);
    gpu_random_sample_locked(d_M_random, k_max, Bk, d_state, d_lock);

    gpu_gather_1d(d_M_current, d_M, d_M_random, k_max);
    cudaMemcpy(d_M_idx, d_M_random, k_max * sizeof(int), cudaMemcpyDeviceToDevice);

    int number_of_blocks = Bk / BLOCK_SIZE;
    if (Bk % BLOCK_SIZE) number_of_blocks++;
    set_all << < number_of_blocks, min(Bk, BLOCK_SIZE) >> > (d_delta_old, -1., Bk);

    std::vector <std::vector<at::Tensor>> R;

    for (int k_idx = 0; k_idx < ks.size(); k_idx++) {
        int k = ks[k_idx];
        for (int l_idx = 0; l_idx < ls.size(); l_idx++) {
            int l = ls[l_idx];

            if (l_idx == 0 & k_idx == 0) {
                cudaMemcpy(d_M_best, d_M_current, k * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_M_idx_best, d_M_idx, k * sizeof(int), cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpy(d_M_current, d_M_best, k * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_M_idx, d_M_idx_best, k * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            int termination_criterion = 0;
            set(d_cost_best, 0, 1000000.);
            cudaMemset(d_termination_criterion, 0, sizeof(float));

            while (termination_criterion < termination_rounds) {

                //// compute L ////
                gpu_compute_L_save(d_L, d_L_sizes_change, d_L_sizes, d_lambda,
                                   d_dist_n_Bk, d_dist_n_Bk_set,
                                   d_delta_old, d_delta,
                                   d_M, d_M_idx,
                                   d_data,
                                   n, d, k);

                //// find dimensions ////
                gpu_find_dimensions_save(d_D, d_Z, d_X, d_H,
                                         d_L, d_L_sizes_change, d_L_sizes, d_lambda,
                                         d_M_current, d_M_idx,
                                         d_data,
                                         n, d, k, l);

                //// assign points /////
                gpu_assign_points(d_C, d_C_sizes,
                                  d_D, d_Ds, d_D_sizes,
                                  d_M_current,
                                  d_data,
                                  n, d, k);

                //// evaluate clustering ////
                gpu_evaluate_cluster(d_cost,
                                     d_C, d_C_sizes,
                                     d_D, d_D_sizes,
                                     d_data,
                                     n, d, k);

                //// update best ////
                termination_criterion += 1;
                gpu_update_best_SAVE(d_M_idx, d_M_idx_best, d_cost, d_cost_best,
                                     d_termination_criterion,
                                     d_M_best, d_M_current,
                                     d_C, d_C_sizes, d_C_best, d_C_sizes_best,
                                     d_bad,
                                     min_deviation, n, k);

                if (termination_criterion >= termination_rounds) {
                    //only read from device version of termination_criterion as few times as possible
                    cudaMemcpy(&termination_criterion, d_termination_criterion, sizeof(int), cudaMemcpyDeviceToHost);
                }

                //replace bad medoids
                gpu_random_sample_locked(d_M_random, k, Bk, d_state, d_lock);
                gpu_replace_medoids_kernel_pre << < 1, 1 >> > (d_M_idx, d_M_idx_best, d_M_current, d_M_random, d_M, Bk,
                        d_M_best, d_bad, k, n);

            }

            //// Refinement Phase ////
            gpu_find_dimensions(d_D, d_Z, d_X,
                                d_C_best, d_C_sizes_best,
                                d_M_best,
                                d_data,
                                n, d, k, l);

            gpu_assign_points(d_C_best, d_C_sizes_best,
                              d_D, d_Ds, d_D_sizes,
                              d_M_best,
                              d_data,
                              n, d, k);

            remove_outliers(d_C_result, d_C_best, d_C_sizes_best,
                            d_D,
                            d_delta,
                            d_M_best,
                            d_data,
                            n, d, k);

            // building result
            std::vector <at::Tensor> r;

            torch::Tensor M_Tensor = torch::zeros({k}, torch::kInt32);
            torch::Tensor D_Tensor = torch::zeros({k, d}, torch::kBool);
            torch::Tensor C_Tensor = torch::zeros({n}, torch::kInt32);

            cudaMemcpy(M_Tensor.data_ptr<int>(), d_M_best, k * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(D_Tensor.data_ptr<bool>(), d_D, k * d * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(C_Tensor.data_ptr<int>(), d_C_result, n * sizeof(int), cudaMemcpyDeviceToHost);

            r.push_back(M_Tensor);
            r.push_back(D_Tensor);
            r.push_back(C_Tensor);

            R.push_back(r);
        }
    }

    // free all
    cudaFree(d_bad);
    cudaFree(d_C);
    cudaFree(d_C_sizes);
    cudaFree(d_C_best);
    cudaFree(d_C_sizes_best);
    cudaFree(d_C_result);
    cudaFree(d_cost);
    cudaFree(d_cost_best);
    cudaFree(d_D);
    cudaFree(d_Ds);
    cudaFree(d_D_sizes);
    cudaFree(d_data);
    cudaFree(d_delta);
    cudaFree(d_delta_old);
    cudaFree(d_dist_n_Bk);
    cudaFree(d_dist_n_Bk_set);
    cudaFree(d_H);
    cudaFree(d_L);
    cudaFree(d_L_sizes);
    cudaFree(d_L_sizes_change);
    cudaFree(d_lambda);
    cudaFree(d_lock);
    cudaFree(d_M);
    cudaFree(d_M_best);
    cudaFree(d_M_current);
    cudaFree(d_M_idx);
    cudaFree(d_M_idx_best);
    cudaFree(d_M_random);
    cudaFree(d_S);
    cudaFree(d_sigma);
    cudaFree(d_state);
    cudaFree(d_termination_criterion);
    cudaFree(d_X);
    cudaFree(d_Z);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
//    cudaProfilerStop();

    return R;
}