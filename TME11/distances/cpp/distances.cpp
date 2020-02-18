#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>



/// Affiche la taille du tenseur
void show_size(std::string const & name, torch::Tensor x) {
  std::cerr << "s(" << name << ")=";
  for(auto i = 0; i < x.dim(); ++i) {
    if (i > 0) std::cerr << "x";
    std::cerr << x.size(i);
  }
  std::cerr  << std::endl;
}


torch::Tensor distances_forward(
    torch::Tensor x,
    torch::Tensor y,
    double p = 2,
    double eps = 1e-6) {
  torch::Tensor output = at::empty({ x.size(0), y.size(0) }, x.options());

  //  TODO:  Implémenter en utilisant at::pairwise_distance

  return output;
}

std::vector<torch::Tensor> distances_backward(
  torch::Tensor grad_output,
  torch::Tensor output,
  torch::Tensor x,
  torch::Tensor y,
  double p=2,
  double eps=1e-6) {

    auto gradx = at::zeros_like(x);
    auto grady = at::zeros_like(y);

    //  TODO:  Implémenter

    return {gradx, grady};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &distances_forward, "Distances forward");
  m.def("backward", &distances_backward, "Distances backward");
}
