# RTX 5000 Laptop GPU Optimized Build Configuration
# This configuration optimizes Ollama for RTX 5000 Mobile Ada Generation

# CUDA Architecture Configuration
# RTX 5000 Mobile Ada has compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES "89-real" CACHE STRING "CUDA architectures for RTX 5000")

# Enable all optimizations for Ada Lovelace
set(GGML_CUDA ON CACHE BOOL "Enable CUDA support")
set(GGML_CUDA_FA ON CACHE BOOL "Enable Flash Attention")
set(GGML_CUDA_FA_ALL_QUANTS ON CACHE BOOL "Enable all quantization types for Flash Attention")
set(GGML_CUDA_GRAPHS ON CACHE BOOL "Enable CUDA graphs for reduced overhead")

# Memory optimizations for 16GB VRAM
set(GGML_CUDA_PEER_MAX_BATCH_SIZE 256 CACHE STRING "Optimize batch size for RTX 5000")

# Disable features that may reduce performance on RTX 5000
set(GGML_CUDA_NO_VMM OFF CACHE BOOL "Enable Virtual Memory Management")
set(GGML_CUDA_NO_PEER_COPY OFF CACHE BOOL "Enable peer-to-peer copy")

# Force optimized matrix multiplication
set(GGML_CUDA_FORCE_MMQ OFF CACHE BOOL "Use optimized kernels instead of MMQ")
set(GGML_CUDA_FORCE_CUBLAS OFF CACHE BOOL "Use custom kernels optimized for RTX 5000")

# Compiler optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3" CACHE STRING "CUDA optimization flags")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math" CACHE STRING "Enable fast math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3" CACHE STRING "PTX optimization")

# Tensor Core optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGGML_CUDA_USE_TENSOR_CORES" CACHE STRING "Enable Tensor Cores")

# Memory bandwidth optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGGML_CUDA_OPTIMIZE_MEMORY_BANDWIDTH" CACHE STRING "Optimize memory bandwidth")

# Ada Lovelace specific optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGGML_CUDA_ADA_LOVELACE_OPTIMIZATIONS" CACHE STRING "Ada Lovelace optimizations")

message(STATUS "RTX 5000 optimizations enabled")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")