#include "gpu_util.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include "curand_kernel.h"

#define SECTION_SIZE 64
#define BLOCK_SIZE 1024

int *gpu_shuffle(int *h_indices, int n) {


    int *h_a = new int[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = h_indices[i];
    }

    for (int i = n - 1; i > 0; i--) {
        int j = std::rand() % (i + 1);
        int tmp_idx = h_a[i];
        h_a[i] = h_a[j];
        h_a[j] = tmp_idx;
    }

    int *d_a;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMemcpy(d_a, h_a, sizeof(int) * n, cudaMemcpyHostToDevice);

    return d_a;
}

void gpu_shuffle_v2(int *d_a, int *h_indices, int n) {
    int *h_a = new int[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = h_indices[i];
    }

    for (int i = n - 1; i > 0; i--) {
        int j = std::rand() % (i + 1);
        int tmp_idx = h_a[i];
        h_a[i] = h_a[j];
        h_a[j] = tmp_idx;
    }
    cudaMemcpy(d_a, h_a, sizeof(int) * n, cudaMemcpyHostToDevice);
}

__global__ void gpu_shuffle_v3_kernel(int *d_in, int n, curandState *state) {
    for (int i = n - 1; i >= 0; i--) {
        int j = curand(&state[i]) % (i + 1);
        int tmp_idx = d_in[i];
        d_in[i] = d_in[j];
        d_in[j] = tmp_idx;
    }
}

void gpu_shuffle_v3(int *d_in, int n, curandState *d_state) {

    //todo use https://stackoverflow.com/questions/12653995/how-to-generate-random-permutations-with-cuda

    gpu_shuffle_v3_kernel << < 1, 1 >> > (d_in, n, d_state);

}

__global__ void gpu_random_sample_kernel(int *d_in, int k, int n, curandState *state) {
    for (int i = 0; i < k; i++) {
        int j = curand(&state[threadIdx.x]) % (n);
        int tmp_idx = d_in[i];
        d_in[i] = d_in[j];
        d_in[j] = tmp_idx;
    }
}

void gpu_random_sample(int *d_in, int k, int n, curandState *d_state) {
    gpu_random_sample_kernel << < 1, 1 >> > (d_in, k, n, d_state);
}

__global__ void gpu_random_sample_kernel_locked(int *d_in, int k, int n, curandState *state, int *d_lock) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        int j = curand(&state[threadIdx.x]) % (n);

        if (i < j) {
            while (atomicCAS(&d_lock[i], 0, 1) != 0);//printf("test1 %d, %d, d_lock[i]: %d\n", i, j, d_lock[i]);
            while (atomicCAS(&d_lock[j], 0, 1) != 0);//printf("test2 %d, %d, d_lock[j]: %d\n", i, j, d_lock[j]);
        } else if (i > j) {
            while (atomicCAS(&d_lock[j], 0, 1) != 0);//printf("test3 %d, %d, d_lock[j]: %d\n", i, j, d_lock[j]);
            while (atomicCAS(&d_lock[i], 0, 1) != 0);//printf("test4 %d, %d, d_lock[i]: %d\n", i, j, d_lock[i]);
        } else {
            while (atomicCAS(&d_lock[i], 0, 1) != 0);//printf("test5 %d, %d, d_lock[i]: %d\n", i, j, d_lock[i]);
        }

        int tmp_idx = d_in[i];
        d_in[i] = d_in[j];
        d_in[j] = tmp_idx;

        if (i < j) {
            atomicExch(&d_lock[j], 0);
            atomicExch(&d_lock[i], 0);
        } else if (i > j) {
            atomicExch(&d_lock[i], 0);
            atomicExch(&d_lock[j], 0);
        } else {
            atomicExch(&d_lock[i], 0);
        }
    }
}


__global__ void gpu_random_sample_kernel_locked_v2(int *d_in, int k, int n, curandState *state, int *d_lock) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {

        int j = curand(&state[threadIdx.x]) % (n);

        if (i > j) {
            int tmp = j;
            j = i;
            i = tmp;
        }

        bool success = false;

        while (!success) {
            if (atomicCAS(&d_lock[i], 0, 1) == 0) {
                if (i == j || atomicCAS(&d_lock[j], 0, 1) == 0) {
                    int tmp_idx = d_in[i];
                    d_in[i] = d_in[j];
                    d_in[j] = tmp_idx;

                    success = true;
                    if (i != j) {
                        atomicExch(&d_lock[j], 0);
                    }
                }
                atomicExch(&d_lock[i], 0);
            }
        }
    }
}

void gpu_random_sample_locked(int *d_in, int k, int n, curandState *d_state, int *d_lock) {
    cudaMemset(d_lock, 0, n * sizeof(int));
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    gpu_random_sample_kernel_locked_v2 << < number_of_blocks, min(k, BLOCK_SIZE) >> > (d_in, k, n, d_state, d_lock);
}

//todo change this to be correct kind of locking
__global__ void gpu_not_random_sample_kernel_locked(int *d_in, int k, int n, int *state, int *d_lock) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        int j = state[threadIdx.x] % (n);
        state[threadIdx.x] += 11;

        if (i < j) {
            while (atomicCAS(&d_lock[i], 0, 1) != 0);
            while (atomicCAS(&d_lock[j], 0, 1) != 0);
        } else if (i > j) {
            while (atomicCAS(&d_lock[j], 0, 1) != 0);
            while (atomicCAS(&d_lock[i], 0, 1) != 0);
        } else {
            while (atomicCAS(&d_lock[i], 0, 1) != 0);
        }

        int tmp_idx = d_in[i];
        d_in[i] = d_in[j];
        d_in[j] = tmp_idx;

        if (i < j) {
            atomicExch(&d_lock[j], 0);
            atomicExch(&d_lock[i], 0);
        } else if (i > j) {
            atomicExch(&d_lock[i], 0);
            atomicExch(&d_lock[j], 0);
        } else {
            atomicExch(&d_lock[i], 0);
        }
    }
}

__global__ void gpu_not_random_sample_kernel_locked_v2(int *d_in, int k, int n, int *state, int *d_lock) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        int j = state[threadIdx.x] % (n);
        state[threadIdx.x] += 11;

        if (i > j) {
            int tmp = j;
            j = i;
            i = tmp;
        }

        bool success = false;

        while (!success) {
            if (atomicCAS(&d_lock[i], 0, 1) == 0) {
                if (i == j || atomicCAS(&d_lock[j], 0, 1) == 0) {
                    int tmp_idx = d_in[i];
                    d_in[i] = d_in[j];
                    d_in[j] = tmp_idx;

                    success = true;
                    if (i != j) {
                        atomicExch(&d_lock[j], 0);
                    }
                }
                atomicExch(&d_lock[i], 0);
            }
        }
    }
}

void gpu_not_random_sample_locked(int *d_in, int k, int n, int *d_state, int *d_lock) {
    cudaMemset(d_lock, 0, n * sizeof(int));
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    gpu_not_random_sample_kernel_locked << < 1, 1 >> > (d_in, k, n, d_state, d_lock);
}

//float *copy_to_flatten_device(float **h_mem, int height, int width) {
//    float *d_mem;
//    cudaMalloc(&d_mem, height * width * sizeof(float));
//    for (int row = 0; row < height; row++) {
//        cudaMemcpy(&d_mem[row * width], h_mem[row], sizeof(float) * width, cudaMemcpyHostToDevice);
//    }
//    return d_mem;
//}

float *copy_to_flatten_device(at::Tensor h_mem, int height, int width) {
    float *d_mem;
    cudaMalloc(&d_mem, height * width * sizeof(float));
//    cudaMalloc(&d_mem, height * width * sizeof(float));
//    float *tmp = new float[height * width];
//    for (int row = 0; row < height; row++) {
//        float *h_mem_row = h_mem[row].data_ptr<float>();
////        cudaMemcpy(&d_mem[row * width], h_mem_row, sizeof(float) * width, cudaMemcpyHostToDevice);
//        for (int col = 0; col < width; col++) {
//            tmp[row * width + col] = h_mem_row[col];
//        }
//    }
    cudaMemcpy(d_mem, h_mem.data<float>(), height * width * sizeof(float), cudaMemcpyHostToDevice);
    return d_mem;
}

__global__
void gpu_gather_2d_kernel(float *d_source, int *d_indices, int height, int width,
                          float *d_result) {//todo change order
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < width; j += blockDim.y * gridDim.y) {
            d_result[i * width + j] = d_source[d_indices[i] * width + j];
        }
    }
}

float *gpu_gather_2d(float *d_source, int *d_indices, int height, int width) {
    float *d_result;
    cudaMalloc(&d_result, height * width * sizeof(float));


    int g_h = height / 32;
    if (height % 32) g_h++;

    int g_w = width / 32;
    if (width % 32) g_w++;

    dim3 grid(g_h, g_w);
    dim3 block(min(32, height), min(32, width));
    gpu_gather_2d_kernel << < grid, block >> > (d_source, d_indices, height, width, d_result);//todo fix size

    return d_result;
}


__global__
void gpu_gather_1d_kernel(int *d_source, int *d_indices, int length,
                          int *d_result) {//todo change order
    for (int j = 0; j < length; j++) {
        d_result[j] = d_source[d_indices[j]];
    }
}

void gpu_gather_1d(int *d_result, int *d_source, int *d_indices, int length) {
    gpu_gather_1d_kernel << < 1, 1 >> > (d_source, d_indices, length, d_result);
}

__global__
void set_kernel(int *x, int i, int value) {
    x[i] = value;
}

void set(int *x, int i, int value) {
    set_kernel << < 1, 1 >> > (x, i, value);
}


__global__
void set_kernel(int *x, int *idx, int i, int value) {
    x[i] = idx[value];
}

void set(int *x, int *idx, int i, int value) {
    set_kernel << < 1, 1 >> > (x, idx, i, value);
}

__global__
void set_kernel(float *x, int i, float value) {
    x[i] = value;
}

void set(float *x, int i, float value) {
    set_kernel << < 1, 1 >> > (x, i, value);
}


__global__
void print_array_gpu_kernel(float *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%f ", (float) x[i]);
    }
    printf("\n");
}

void print_array_gpu(float *d_X, int n) {
    print_array_gpu_kernel << < 1, 1 >> > (d_X, n);
    cudaDeviceSynchronize();
}

__global__
void print_array_gpu_kernel(int *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%d ", (int) x[i]);
    }
    printf("\n");
}

void print_array_gpu(int *d_X, int n) {
    print_array_gpu_kernel << < 1, 1 >> > (d_X, n);
    cudaDeviceSynchronize();
}

__global__
void print_array_gpu_kernel(bool *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i]) {
            printf("true ");
        } else {
            printf("false ");
        }
    }
    printf("\n");
}

void print_array_gpu(bool *d_X, int n) {
    print_array_gpu_kernel << < 1, 1 >> > (d_X, n);
    cudaDeviceSynchronize();
}
//
//void print_array_gpu(bool *d_X, int n, int m) {
//    for(int i =0;i<n;i++) {
//        print_array_gpu_kernel << < 1, 1 >> > (&d_X[i*m], m);
//        cudaDeviceSynchronize();
//    }
//}

__global__
void print_array_gpu_kernel(float *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%f ", (float) x[i]);
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void print_array_gpu(float *d_X, int n, int m) {
    print_array_gpu_kernel << < 1, 1 >> > (d_X, n, m);
    cudaDeviceSynchronize();
}

__global__
void print_array_gpu_kernel(bool *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        if (x[i]) {
            printf("true ");
        } else {
            printf("false ");
        }
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }
    printf("\n");
}


void print_array_gpu(bool *d_X, int n, int m) {
    print_array_gpu_kernel << < 1, 1 >> > (d_X, n, m);
    cudaDeviceSynchronize();
}

__global__
void init_seed(curandState *state, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__
void gpu_clone_kernel(int *d_to, int *d_from, int size) {
    for (int i = 0; i < size; i++) {
        d_to[i] = d_from[i];
    }
}

void gpu_clone(int *d_to, int *d_from, int size) {
    gpu_clone_kernel << < 1, 1 >> > (d_to, d_from, size);
}


__global__
void set_all(float *d_X, float value, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d_X[i] = value;
    }
}

__global__
void set_all(int *d_X, int value, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d_X[i] = value;
    }
}

__global__
void set_all(bool *d_X, bool value, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d_X[i] = value;
    }
}


__global__
void scan_kernel_eff(int *x, int *y, int n) {
/**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }
}


__global__
void scan_kernel_eff_large1(int *x, int *y, int *S, int n) {
/**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        S[blockIdx.x] = XY[SECTION_SIZE - 1];
    }

}

__global__
void scan_kernel_eff_large3(int *y, int *S, int n) {
/**
 * from the cuda book
 */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n) {
        y[i] += S[blockIdx.x - 1];
    }
}

void inclusive_scan(int *source, int *result, int n) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE) numBlocks++;

    if (n > SECTION_SIZE) {
        int *S;
        cudaMalloc((void **) &S, numBlocks * sizeof(int));
        scan_kernel_eff_large1 << < numBlocks, SECTION_SIZE >> > (source, result, S, n);
        inclusive_scan(S, S, numBlocks);
        scan_kernel_eff_large3 << < numBlocks, SECTION_SIZE >> > (result, S, n);
        cudaFree(S);
    } else {
        scan_kernel_eff << < numBlocks, SECTION_SIZE >> > (source, result, n);
    }
}

int *device_allocate_int(int n) {
    int *tmp;
    cudaMalloc(&tmp, n * sizeof(int));
    return tmp;
}

float *device_allocate_float(int n) {
    float *tmp;
    cudaMalloc(&tmp, n * sizeof(float));
    return tmp;
}

bool *device_allocate_bool(int n) {
    bool *tmp;
    cudaMalloc(&tmp, n * sizeof(bool));
    return tmp;
}

int *device_allocate_int_zero(int n) {
    int *tmp;
    cudaMalloc(&tmp, n * sizeof(int));
    cudaMemset(tmp, 0, n * sizeof(int));
    return tmp;
}

float *device_allocate_float_zero(int n) {
    float *tmp;
    cudaMalloc(&tmp, n * sizeof(float));
    cudaMemset(tmp, 0, n * sizeof(float));
    return tmp;
}

bool *device_allocate_bool_zero(int n) {
    bool *tmp;
    cudaMalloc(&tmp, n * sizeof(bool));
    cudaMemset(tmp, 0, n * sizeof(bool));
    return tmp;
}