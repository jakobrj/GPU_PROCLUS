#ifndef PROCLUS_GPU_UTIL_H
#define PROCLUS_GPU_UTIL_H

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>
#include <cstdlib>

float *compute_l2_norm_to_medoid(float **S, int m_i, int n, int d);

float *compute_l2_norm_to_medoid(at::Tensor data, int *S, int m_i, int n, int d);

float *compute_l2_norm_to_medoid(at::Tensor data, int m_i, int n, int d);

float *compute_l1_norm_to_medoid(float **S, int m_i, bool *D_i, int n, int d);

float *compute_l1_norm_to_medoid(at::Tensor S, int m_i, bool *D_i, int n, int d);


void compute_l2_norm_to_medoid(float *dist, at::Tensor data, int m_i, int n, int d);

void compute_l2_norm_to_medoid(float *dist, at::Tensor data, int *S, int m_i, int n, int d);


int argmax_1d(float *values, int n);

template<typename T>
extern int argmin_1d(T *values, int n);

std::pair<int, int> *argmin_2d(float **values, int n, int m);

void index_wise_minimum(float *values_1, float *values_2, int n);

float mean_1d(float *values, int n);

bool all_close_1d(float *values_1, float *values_2, int n);

bool all_close_2d(float **values_1, float **values_2, int n, int m);

bool close(float value_1, float value_2);

int *shuffle(int *indices, int n);

int *random_sample(int *indices, int k, int n);

int *not_random_sample(int *in, int *state, int state_length, int k, int n);

template<typename T>
T **array_2d(int n, int m);

template<typename T>
T **zeros_2d(int n, int m);

template<typename T>
T *zeros_1d(int n);

float **gather_2d(float **S, int *indices, int k, int d);

float **gather_2d(at::Tensor S, int *indices, int k, int d);

int *fill_with_indices(int n);

void print_debug(char *str, bool debug);

void print_array(float *x, int n);

void print_array(int *x, int n);

void print_array(bool *x, int n);

void print_array(int **X, int n, int m);

void print_array(float **X, int n, int m);

void print_array(bool **X, int n, int m);

#endif //PROCLUS_GPU_UTIL_H
