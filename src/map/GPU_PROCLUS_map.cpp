#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "../utils/util.h"
#include "../utils/gpu_util.cuh"
#include "../algorithms/PROCLUS.h"
#include "../algorithms/GPU_PROCLUS.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("PROCLUS",    &PROCLUS,    "");
m.def("PROCLUS_KEEP",    &PROCLUS_KEEP,    "");
m.def("PROCLUS_SAVE",    &PROCLUS_SAVE,    "");
m.def("PROCLUS_PARAM",    &PROCLUS_PARAM,    "");
m.def("GPU_PROCLUS",    &GPU_PROCLUS,    "");
m.def("GPU_PROCLUS_KEEP",    &GPU_PROCLUS_KEEP,    "");
m.def("GPU_PROCLUS_SAVE",    &GPU_PROCLUS_SAVE,    "");
m.def("GPU_PROCLUS_PARAM",    &GPU_PROCLUS_PARAM,    "");
m.def("GPU_PROCLUS_PARAM_2",    &GPU_PROCLUS_PARAM_2,    "");
m.def("GPU_PROCLUS_PARAM_3",    &GPU_PROCLUS_PARAM_3,    "");
}