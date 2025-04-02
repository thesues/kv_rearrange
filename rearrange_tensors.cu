// rearrange_tensors.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T>
__global__ void rearrange_kernel(
    const T* __restrict__ t1,
    T* __restrict__ t2,
    int N, int B, int H, int C,
    int d,
    int tensor_subset_size,
    int block_size,
    int token_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * B * H * C;
    if (idx >= total_elements) return;

    int curr_n = idx / block_size;
    int curr_b = (idx / token_size) % B;
    int curr_h = (idx / C) % H;
    int curr_c = idx % C;

    int tp_group = curr_h * d / H;
    int dst_h = curr_h % (H / d);

    int tp_group_offset = curr_n * (block_size / d)
                        + curr_b * (H / d) * C
                        + dst_h * C
                        + curr_c;

    int dst_pos = tensor_subset_size * tp_group + tp_group_offset;

    t2[dst_pos] = t1[idx];
}

// C++ 接口模板函数
template <typename T>
void launch_rearrange_tensors(torch::Tensor t1, torch::Tensor t2, int d) {
    int N = t1.size(0);
    int B = t1.size(1);
    int H = t1.size(2);
    int C = t1.size(3);

    int total_elements = N * B * H * C;
    int block_size = B * H * C;
    int token_size = H * C;
    int tensor_subset_size = total_elements / d;

    const int threads = 1024;
    const int blocks = (total_elements + threads - 1) / threads;

    rearrange_kernel<T><<<blocks, threads>>>(
        t1.data_ptr<T>(),
        t2.data_ptr<T>(),
        N, B, H, C, d,
        tensor_subset_size,
        block_size,
        token_size
    );
}

void rearrange_tensors_cuda(torch::Tensor t1, torch::Tensor t2, int d) {
    TORCH_CHECK(t1.device().is_cuda(), "t1 must be a CUDA tensor");
    // TORCH_CHECK(t2.device().is_cuda(), "t2 must be a CUDA tensor");
    TORCH_CHECK(t1.sizes() == t2.sizes(), "t1 and t2 must have the same shape");
    int H = t1.size(2);
    TORCH_CHECK(H % d == 0, "H must be divisible by d");


    //FIXME: replaced by AT_DISPATCH_FLOATING_TYPES
    if (t1.scalar_type() == torch::kFloat32) {
        launch_rearrange_tensors<float>(t1, t2, d);
    } else if (t1.scalar_type() == torch::kFloat16) {
        launch_rearrange_tensors<at::Half>(t1, t2, d);
    }else if (t1.scalar_type() == torch::kInt) {
        launch_rearrange_tensors<int>(t1, t2, d);
    } else {
        TORCH_CHECK(false, "Unsupported tensor type");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rearrange_tensors_cuda", &rearrange_tensors_cuda, "Rearrange Tensors (CUDA)");
}