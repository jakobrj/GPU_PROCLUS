//
// Created by jakobrj on 3/8/21.
//

#ifndef PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH
#define PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH

std::pair<std::vector<at::Tensor>, int>
GPU_PROCLUS(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds, bool debug);

std::pair<std::vector<at::Tensor>, int>
GPU_PROCLUS_KEEP(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds,
                 bool debug);

std::pair<std::vector<at::Tensor>, int>
GPU_PROCLUS_SAVE(at::Tensor data, int k, int l, float a, float b, float min_deviation, int termination_rounds,
                 bool debug);

std::pair<std::vector <std::vector<at::Tensor>>, int>
GPU_PROCLUS_PARAM(at::Tensor data, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
                  int termination_rounds);

std::pair<std::vector <std::vector<at::Tensor>>, int>
GPU_PROCLUS_PARAM_2(at::Tensor data, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
                  int termination_rounds);

std::pair<std::vector <std::vector<at::Tensor>>, int>
GPU_PROCLUS_PARAM_3(at::Tensor data, std::vector<int> ks, std::vector<int> ls, float a, float b, float min_deviation,
                  int termination_rounds);

#endif //PROCLUS_GPU_GPU_PROCLUS_SAVE_CUH
