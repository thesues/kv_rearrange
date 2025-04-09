// rearrange_tensors.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>


// template <typename T>
// __global__ void rearrange_kernel(
//     const T* __restrict__ t1_ptr, 
//     T* __restrict__ t2_ptr,
//     int N, int B, int H, int C, int d,
//     int tensor_subset_size,
//     int block_size,
//     int token_size)
// {
//     int BLOCK_SIZE = blockDim.x;
//     int pid = blockIdx.x;
//     int tid = threadIdx.x;

//     int offset = pid * BLOCK_SIZE + tid;

//     if (offset >= N * B * H * C) return;

//     int curr_n = offset / block_size;
//     int curr_b = (offset / token_size) % B;
//     int curr_h = (offset / C) % H;
//     int curr_c = offset % C;

//     int src_pos = offset;

//     int tp_group = curr_h * d / H;
//     int dst_h = curr_h % (H / d);
//     int tp_group_offset = curr_n * (block_size / d) + curr_b * (H / d) * C + dst_h * C + curr_c;

//     int dst_pos = tensor_subset_size * tp_group + tp_group_offset;
    

//     t2_ptr[dst_pos] = t1_ptr[src_pos];
// }


template <typename T>
__global__ void rearrange_kernel_optimized(
    const T* __restrict__ t1_ptr, 
    T* __restrict__ t2_ptr,
    int N, int B, int H, int C, int d,
    int tensor_subset_size,
    int block_size,
    int token_size,
    int h_div_d,
    int block_size_div_d
) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = N * B * H * C;

    if (offset >= total_elems) return;

    // 使用预计算除法减少开销
    int curr_n = offset / block_size;
    int tmp = offset % block_size;
    int curr_b = tmp / token_size;
    tmp = tmp % token_size;
    int curr_h = tmp / C;
    int curr_c = tmp % C;

    // 目标位置计算（group-based rearrangement）
    int tp_group = (curr_h * d) / H;     // group id
    int dst_h = curr_h % h_div_d;        // head 内部 offset
    int tp_group_offset = curr_n * block_size_div_d
                        + curr_b * h_div_d * C
                        + dst_h * C
                        + curr_c;

    int dst_pos = tensor_subset_size * tp_group + tp_group_offset;

    // 内存访问：读取是顺序的，写入虽然 rearranged，但偏移整体有规整结构
    t2_ptr[dst_pos] = t1_ptr[offset];
}




// C++ 接口模板函数
template <typename T>
void launch_rearrange_tensors(torch::Tensor t1, torch::Tensor t2, int d) {
    // int N = t1.size(0);
    // int B = t1.size(1);
    // int H = t1.size(2);
    // int C = t1.size(3);

    // int total_elements = N * B * H * C;
    // int block_size = B * H * C;
    // int token_size = H * C;
    // int tensor_subset_size = total_elements / d;

    // const int threads = 1024;
    // const int blocks = (total_elements + threads - 1) / threads;

    // rearrange_kernel<T><<<blocks, threads>>>(
    //     t1.data_ptr<T>(),
    //     t2.data_ptr<T>(),
    //     N, B, H, C, d,
    //     tensor_subset_size,
    //     block_size,
    //     token_size
    // );

    int N = t1.size(0);
    int B = t1.size(1);
    int H = t1.size(2);
    int C = t1.size(3);

    int total_elements = N * B * H * C;
    int block_size = B * H * C;
    int tensor_subset_size = total_elements / d;
    int token_size = H * C;


    int threads_per_block = 1024;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    int h_div_d = H / d;
    int block_size_div_d = block_size / d;

    rearrange_kernel_optimized<T><<<blocks, threads_per_block>>>(
        t1.data_ptr<T>(),
        t2.data_ptr<T>(),
        N, B, H, C, d,
        tensor_subset_size,
        block_size,
        token_size,
        h_div_d,
        block_size_div_d
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