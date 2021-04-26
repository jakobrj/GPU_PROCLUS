#ifndef PROCLUS_GPU_PROCLUS_H
#define PROCLUS_GPU_PROCLUS_H

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

std::vector <at::Tensor>
PROCLUS(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::vector <at::Tensor>
PROCLUS_KEEP(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::vector <at::Tensor>
PROCLUS_SAVE(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::vector <std::vector <at::Tensor>>
PROCLUS_PARAM(at::Tensor data, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
              int termination_rounds, bool debug);
#endif //PROCLUS_GPU_PROCLUS_H
