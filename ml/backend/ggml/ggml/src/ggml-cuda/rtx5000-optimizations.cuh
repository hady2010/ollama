#pragma once

#include "common.cuh"

// RTX 5000 Mobile Ada Generation Optimizations
// Compute Capability 8.9, 9728 CUDA cores, 304 Tensor cores, 16GB GDDR6

#define RTX5000_CUDA_CORES 9728
#define RTX5000_TENSOR_CORES 304
#define RTX5000_MEMORY_GB 16
#define RTX5000_MEMORY_BANDWIDTH_GBS 576
#define RTX5000_COMPUTE_CAPABILITY 89

// Optimal thread block sizes for RTX 5000
#define RTX5000_OPTIMAL_BLOCK_SIZE_X 32
#define RTX5000_OPTIMAL_BLOCK_SIZE_Y 16
#define RTX5000_OPTIMAL_BLOCK_SIZE_Z 4

// Memory coalescing optimizations
#define RTX5000_MEMORY_ALIGNMENT 128
#define RTX5000_CACHE_LINE_SIZE 128

// Tensor Core utilization macros
#if __CUDA_ARCH__ >= 890
#define RTX5000_USE_TENSOR_CORES 1
#define RTX5000_TENSOR_CORE_M 16
#define RTX5000_TENSOR_CORE_N 16
#define RTX5000_TENSOR_CORE_K 16
#else
#define RTX5000_USE_TENSOR_CORES 0
#endif

// Optimized warp-level primitives for Ada Lovelace
template<typename T>
__device__ __forceinline__ T rtx5000_warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T rtx5000_warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Memory bandwidth optimized copy functions
template<typename T>
__device__ __forceinline__ void rtx5000_vectorized_copy(T* dst, const T* src, int n) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Use 128-bit loads/stores for optimal memory bandwidth
    if (sizeof(T) == 4) { // float
        float4* dst4 = reinterpret_cast<float4*>(dst);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        const int n4 = n / 4;
        
        for (int i = tid; i < n4; i += stride) {
            dst4[i] = src4[i];
        }
    } else if (sizeof(T) == 2) { // half
        float4* dst4 = reinterpret_cast<float4*>(dst);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        const int n8 = n / 8;
        
        for (int i = tid; i < n8; i += stride) {
            dst4[i] = src4[i];
        }
    }
}

// Optimized matrix multiplication for RTX 5000
template<typename T>
__global__ void rtx5000_optimized_gemm(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    float alpha, float beta
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int row = by * RTX5000_OPTIMAL_BLOCK_SIZE_Y + ty;
    const int col = bx * RTX5000_OPTIMAL_BLOCK_SIZE_X + tx;
    
    __shared__ T As[RTX5000_OPTIMAL_BLOCK_SIZE_Y][RTX5000_OPTIMAL_BLOCK_SIZE_X];
    __shared__ T Bs[RTX5000_OPTIMAL_BLOCK_SIZE_X][RTX5000_OPTIMAL_BLOCK_SIZE_Y];
    
    T sum = 0.0f;
    
    for (int k = 0; k < K; k += RTX5000_OPTIMAL_BLOCK_SIZE_X) {
        // Load tiles into shared memory with coalesced access
        if (row < M && (k + tx) < K) {
            As[ty][tx] = A[row * K + k + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((k + ty) < K && col < N) {
            Bs[tx][ty] = B[(k + ty) * N + col];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < RTX5000_OPTIMAL_BLOCK_SIZE_X; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    // Write result with alpha/beta scaling
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Flash Attention optimization for RTX 5000
template<typename T>
__global__ void rtx5000_flash_attention(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int seq_len, int head_dim,
    float scale
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int seq_idx = blockIdx.y;
    
    extern __shared__ T shared_mem[];
    T* shared_K = shared_mem;
    T* shared_V = shared_K + head_dim;
    T* shared_scores = shared_V + head_dim;
    
    // Load Q vector for this sequence position
    T q_vec[16]; // Assuming head_dim <= 16 * 32 = 512
    if (tid < head_dim) {
        q_vec[tid] = Q[bid * seq_len * head_dim + seq_idx * head_dim + tid];
    }
    
    T max_score = -INFINITY;
    T sum_exp = 0.0f;
    T output[16] = {0.0f};
    
    // Process K,V in chunks for memory efficiency
    for (int k_start = 0; k_start < seq_len; k_start += RTX5000_OPTIMAL_BLOCK_SIZE_X) {
        int k_end = min(k_start + RTX5000_OPTIMAL_BLOCK_SIZE_X, seq_len);
        
        // Load K chunk
        for (int k_idx = k_start; k_idx < k_end; ++k_idx) {
            if (tid < head_dim) {
                shared_K[(k_idx - k_start) * head_dim + tid] = 
                    K[bid * seq_len * head_dim + k_idx * head_dim + tid];
                shared_V[(k_idx - k_start) * head_dim + tid] = 
                    V[bid * seq_len * head_dim + k_idx * head_dim + tid];
            }
        }
        __syncthreads();
        
        // Compute attention scores
        for (int k_idx = k_start; k_idx < k_end; ++k_idx) {
            T score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_vec[d] * shared_K[(k_idx - k_start) * head_dim + d];
            }
            score *= scale;
            
            shared_scores[k_idx - k_start] = score;
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
        
        // Compute softmax and accumulate output
        for (int k_idx = k_start; k_idx < k_end; ++k_idx) {
            T exp_score = expf(shared_scores[k_idx - k_start] - max_score);
            sum_exp += exp_score;
            
            for (int d = 0; d < head_dim; ++d) {
                output[d] += exp_score * shared_V[(k_idx - k_start) * head_dim + d];
            }
        }
        __syncthreads();
    }
    
    // Normalize and write output
    if (tid < head_dim) {
        O[bid * seq_len * head_dim + seq_idx * head_dim + tid] = output[tid] / sum_exp;
    }
}

// Memory pool management for RTX 5000
class RTX5000MemoryPool {
private:
    void* pool_ptr;
    size_t pool_size;
    size_t allocated_size;
    
public:
    RTX5000MemoryPool(size_t size = RTX5000_MEMORY_GB * 1024 * 1024 * 1024 * 0.8) {
        pool_size = size;
        allocated_size = 0;
        cudaMalloc(&pool_ptr, pool_size);
    }
    
    ~RTX5000MemoryPool() {
        cudaFree(pool_ptr);
    }
    
    void* allocate(size_t size) {
        size = (size + RTX5000_MEMORY_ALIGNMENT - 1) & ~(RTX5000_MEMORY_ALIGNMENT - 1);
        if (allocated_size + size > pool_size) {
            return nullptr;
        }
        void* ptr = static_cast<char*>(pool_ptr) + allocated_size;
        allocated_size += size;
        return ptr;
    }
    
    void reset() {
        allocated_size = 0;
    }
    
    size_t available() const {
        return pool_size - allocated_size;
    }
};

// Launch configuration helpers
inline dim3 rtx5000_optimal_grid_size(int total_threads) {
    int blocks = (total_threads + RTX5000_OPTIMAL_BLOCK_SIZE_X - 1) / RTX5000_OPTIMAL_BLOCK_SIZE_X;
    return dim3(min(blocks, RTX5000_CUDA_CORES / RTX5000_OPTIMAL_BLOCK_SIZE_X));
}

inline dim3 rtx5000_optimal_block_size() {
    return dim3(RTX5000_OPTIMAL_BLOCK_SIZE_X, RTX5000_OPTIMAL_BLOCK_SIZE_Y);
}

// Performance monitoring
struct RTX5000PerfCounters {
    float memory_bandwidth_utilization;
    float compute_utilization;
    float tensor_core_utilization;
    int active_warps;
    int memory_transactions;
};

__device__ RTX5000PerfCounters rtx5000_get_perf_counters() {
    RTX5000PerfCounters counters = {};
    // Implementation would use CUDA profiling APIs
    return counters;
}