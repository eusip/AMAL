#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor distances_cuda_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    double p,
    double eps);

std::vector<torch::Tensor> distances_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor output,
    torch::Tensor x1,
    torch::Tensor x2,
    double p,
    double eps);
