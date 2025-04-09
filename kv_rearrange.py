import torch
import triton
import triton.language as tl
import rearrange_cuda 
import time


@triton.jit
def rearrange_kernel_read(
    t1_ptr,
    t2_ptr,
    N,
    B,
    H,
    C,
    d,
    tensor_subset_size,
    block_size,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    curr_n = offsets // block_size
    curr_b = offsets // token_size % B
    curr_h = offsets // C % H
    curr_c = offsets % C

    src_pos = offsets

    tp_group = curr_h * d // H
    dst_h = curr_h % (H // d)
    tp_group_offset = curr_n * (block_size // d) + curr_b * (H // d) * C + dst_h * C + curr_c

    dst_pos = tensor_subset_size * tp_group + tp_group_offset

    tl.store(t1_ptr + src_pos, tl.load(t2_ptr + dst_pos))

@triton.jit
def rearrange_kernel_write(
    t1_ptr,
    t2_ptr,
    N,
    B,
    H,
    C,
    d,
    tensor_subset_size,
    block_size,
    token_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * B * H * C)

    curr_n = offsets // block_size
    curr_b = offsets // token_size % B
    curr_h = offsets // C % H
    curr_c = offsets % C

    src_pos = offsets

    tp_group = curr_h * d // H
    dst_h = curr_h % (H // d)
    tp_group_offset = curr_n * (block_size // d) + curr_b * (H // d) * C + dst_h * C + curr_c

    dst_pos = tensor_subset_size * tp_group + tp_group_offset

    #FIXME: This is a hack to avoid the issue with the compiler
    src_pos = curr_n * block_size + curr_b * token_size + curr_h * C + curr_c

    tl.store(t2_ptr + dst_pos, tl.load(t1_ptr + src_pos, mask), mask=mask)




def rearrange_tensors(t1: torch.Tensor, t2: torch.Tensor, d: int, direction: str):
    N, B, H, C = t1.shape

    assert t2.shape == (N, B, H, C), "Destination tensor must have same shape as source"
    assert H % d == 0, "H must be divisible by d"

    block_size = B * H * C
    token_size = H * C
    tensor_size = N * block_size
    tensor_subset_size = tensor_size // d

    BLOCK_SIZE = 1024
    grid = ((N * B * H * C + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    if direction == "read":
        rearrange_kernel_read[grid](
            t1, t2,
            N, B, H, C,
            d,
            tensor_subset_size,
            block_size,
            token_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    elif direction == "write":
        rearrange_kernel_write[grid](
            t1, t2,
            N, B, H, C,
            d,
            tensor_subset_size,
            block_size,
            token_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError(f"Invalid direction: {direction}")

def rearrange_tensors_to_dram(t1: torch.Tensor, t2: torch.Tensor, d: int):
    N, B, H, C = t1.shape
    assert t2.shape == (N, B, H, C), "Destination tensor must have the same shape as source"
    assert H % d == 0, "H must be divisible by d"
    assert t1.device.type == "cuda", "t1 must be on GPU"
    # assert t2.device.type == "cpu", "t2 must be on CPU"

    heads_per_group = H // d

    t2.copy_(t1.reshape(N, B, d, heads_per_group, C)
               .permute(2, 0, 1, 3, 4)
               .reshape(N, B, H, C))



def test():
    N, B, H, C, d = 1, 2, 4, 3, 2
    t1 = torch.arange(N * B * H * C, device="cuda", dtype=torch.float16).reshape(N, B, H, C)
    print(t1.dtype)
    t2 = torch.empty_like(t1)

    rearrange_tensors(t1, t2, d, "write")
    print(t2)


    t3 = torch.empty_like(t1, device="cuda")
    rearrange_cuda.rearrange_tensors_cuda(t1, t3, d)
    print(t3)


    t4 = torch.empty_like(t1, device="cpu")
    # print(t1.device)
    # print(t2.device)
    rearrange_tensors_to_dram(t1, t4, d)
    print(t4)


def benchmark():
    N, B, H, C, d = 512, 16, 32, 4096, 2
    t1 = torch.arange(N * B * H * C, device="cuda", dtype=torch.float16).reshape(N, B, H, C)

    torch.cuda.synchronize()
    t2 = torch.empty_like(t1)
    start = time.perf_counter()
    rearrange_cuda.rearrange_tensors_cuda(t1, t2, d)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"CUDA rearrange took {end - start:.4f} seconds")
    del t2



    t3 = torch.empty_like(t1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    rearrange_tensors(t1, t3, d, "write")
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Triton rearrange took {end - start:.4f} seconds")
    del t3


    t4 = torch.empty_like(t1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    rearrange_tensors_to_dram(t1, t4, d)
    end = time.perf_counter()
    print(f"pytorch rearrange took {end - start:.4f} seconds")



#warmup
for i in range(10):
    test()

benchmark()