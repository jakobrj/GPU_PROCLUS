#ifndef PROCLUS_GPU_GPU_UTIL_H
#define PROCLUS_GPU_GPU_UTIL_H

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "curand_kernel.h"

void print_array_gpu(int *d_X, int n);

void print_array_gpu(float *d_X, int n);

void print_array_gpu(bool *d_X, int n);

void print_array_gpu(bool *d_X, int n, int m);

int *gpu_shuffle(int *h_indices, int n);

void gpu_shuffle_v2(int *d_a, int *h_indices, int n);

void gpu_shuffle_v3(int *d_in, int n, curandState *d_state);

void gpu_random_sample(int *d_in, int k, int n, curandState *d_state);

void gpu_random_sample_locked(int *d_in, int k, int n, curandState *d_state, int *d_lock);

void gpu_not_random_sample_locked(int *d_in, int k, int n, int *d_state, int *d_lock);

//float *copy_to_flatten_device(float **h_mem, int height, int width);

float *copy_to_flatten_device(at::Tensor h_mem, int height, int width);

float *gpu_gather_2d(float *d_source, int *d_indices, int height, int width);

void gpu_gather_1d(int *d_result, int *d_source, int *d_indices, int length);

void set(int *x, int i, int value);

void set(int *x, int *idx, int i, int value);

void set(float *x, int i, float value);

__global__
void init_seed(curandState *state, int seed);

void gpu_clone(int *d_to, int *d_from, int size);


__global__
void set_all(float *d_X, float value, int n);

__global__
void set_all(int *d_X, int value, int n);

__global__
void set_all(bool *d_X, bool value, int n);

void inclusive_scan(int *source, int *result, int n);


int *device_allocate_int(int n);

float *device_allocate_float(int n);

bool *device_allocate_bool(int n);

int *device_allocate_int_zero(int n);

float *device_allocate_float_zero(int n);

bool *device_allocate_bool_zero(int n);

int get_total_allocation_count();

void add_total_allocation_count(int n);

void reset_total_allocation_count();

#endif //PROCLUS_GPU_GPU_UTIL_H
