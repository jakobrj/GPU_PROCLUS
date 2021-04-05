//
// Created by jakobrj on 3/8/21.
//

#ifndef PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH
#define PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH

std::vector <at::Tensor>
GPU_PROCLUS(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::vector <at::Tensor>
GPU_PROCLUS_KEEP(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::vector <at::Tensor>
GPU_PROCLUS_SAVE(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::vector <std::vector<at::Tensor>>
GPU_PROCLUS_PARAM(at::Tensor data, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
                  int termination_rounds);

#endif //PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH
