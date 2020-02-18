#include "distances_cuda.h"

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor distances_forward(
    torch::Tensor x1,
    torch::Tensor x2,
    double p = 2,
    double eps = 1e-6) {
  CHECK_INPUT(x1);
  CHECK_INPUT(x2);

  return distances_cuda_forward(x1, x2, p, eps);
}

std::vector<torch::Tensor> distances_backward(
  torch::Tensor grad_output,
  torch::Tensor output,
  torch::Tensor x1,
  torch::Tensor x2,
  double p=2,
  double eps=1e-6) {
    CHECK_CUDA(grad_output);
    CHECK_INPUT(output);
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);

  return distances_cuda_backward(grad_output, output, x1, x2, p, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &distances_forward, "distance forward (CUDA)");
  m.def("backward", &distances_backward, "distance backward (CUDA)");
}
