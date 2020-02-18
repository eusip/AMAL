#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "distances_cuda.h"


namespace {

/// Renvoie le nombre de blocs nécessaires
dim3 getblocks(dim3 threads, int Nx, int Ny) {
  auto Bx = Nx / threads.x;
  if (Nx % Bx != 0) Bx += 1;

  auto By = Ny / threads.y;
  if (Ny % By != 0) By += 1;

  return dim3(Bx, By);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t d_pow(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

}



// ---- Forward

namespace {

// CUDA kernel
template <typename scalar_t>
__global__ void distances_cuda_forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t,2> x,
    const torch::PackedTensorAccessor64<scalar_t,2> y,
    torch::PackedTensorAccessor64<scalar_t,2> output,
    double p,
    double eps) {

  // Récupère les index de la thread du GPU
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Calcul de la distance
  if (i < x.size(0) && j < x.size(1)) {
    //  TODO: 
  }
}

} // anonymous namespace



torch::Tensor distances_cuda_forward(
  torch::Tensor x,
  torch::Tensor y,
  double p = 2,
  double eps = 1e-6)
{
  torch::Tensor output = at::empty({ x.size(0), y.size(0) }, x.options());
  const auto state_size = x.size(1);

  const dim3 threads(32, 32);
  const dim3 blocks = getblocks(threads, x.size(0), y.size(0));

  AT_DISPATCH_FLOATING_TYPES(x.type(), "distances_cuda_forward", ([&] {
    distances_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      x.packed_accessor64<scalar_t,2>(),
      y.packed_accessor64<scalar_t,2>(),
      output.packed_accessor64<scalar_t,2>(),
      p, eps
    );
  }));

  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}


// ---- Backward

namespace {
  //  TODO:  Compléter
}

std::vector<torch::Tensor> distances_cuda_backward(
  torch::Tensor grad_output,
  torch::Tensor output,
  torch::Tensor x,
  torch::Tensor y,
  double p=2,
  double eps=1e-6) {

  auto gradx = at::empty_like(x);
  auto grady = at::empty_like(y);


  AT_DISPATCH_FLOATING_TYPES(x.type(), "distances_cuda_backward", ([&] {
    //  TODO:  Compléter
  }));

  AT_CUDA_CHECK(cudaGetLastError());
  return {gradx, grady};
}
