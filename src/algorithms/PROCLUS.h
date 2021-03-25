#ifndef PROCLUS_GPU_PROCLUS_H
#define PROCLUS_GPU_PROCLUS_H

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

std::vector <at::Tensor>
PROCLUS(at::Tensor X, int k, int l, float a, float b, float min_deviation, int termination_rounds);

#endif //PROCLUS_GPU_PROCLUS_H
