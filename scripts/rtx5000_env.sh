#!/bin/bash
# RTX 5000 Laptop GPU Environment Configuration for Ollama
# Optimizes environment variables for maximum performance

echo "Configuring environment for RTX 5000 Mobile Ada Generation..."

# CUDA Configuration
export CUDA_VISIBLE_DEVICES=0  # Use primary GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483647  # 2GB cache

# Memory Management
export CUDA_MALLOC_HEAP_SIZE=2147483648  # 2GB heap
export CUDA_STACK_SIZE=8192
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Performance Optimizations
export CUDA_AUTO_BOOST=1
export CUDA_FORCE_PTX_JIT=1
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0

# GGML CUDA Optimizations
export GGML_CUDA_ENABLE=1
export GGML_CUDA_FORCE_DMMV=0  # Use optimized kernels
export GGML_CUDA_FORCE_MMQ=0   # Use optimized matrix multiplication
export GGML_CUDA_FORCE_CUBLAS=0  # Use custom kernels
export GGML_CUDA_NO_PEER_COPY=0  # Enable peer copy
export GGML_CUDA_FA_ALL_QUANTS=1  # Enable all quantization types

# Memory Pool Configuration (80% of 16GB = 12.8GB)
export GGML_CUDA_POOL_SIZE=13743895347  # ~12.8GB in bytes

# Batch Size Optimization for RTX 5000
export GGML_CUDA_PEER_MAX_BATCH_SIZE=256
export GGML_CUDA_MAX_BATCH_SIZE=512

# Flash Attention Optimizations
export GGML_CUDA_FA=1
export GGML_CUDA_FA_TILE_SIZE=32
export GGML_CUDA_FA_BLOCK_SIZE=256

# Tensor Core Utilization
export GGML_CUDA_USE_TENSOR_CORES=1
export GGML_CUDA_TENSOR_CORE_PRECISION=fp16

# Memory Bandwidth Optimizations
export GGML_CUDA_MEMORY_POOL_TYPE=unified
export GGML_CUDA_PREFETCH_ENABLED=1
export GGML_CUDA_ASYNC_COPY=1

# Compute Stream Configuration
export GGML_CUDA_STREAMS=4
export GGML_CUDA_STREAM_PRIORITY=high

# Ollama Specific Optimizations
export OLLAMA_GPU_OVERHEAD=0.1  # 10% overhead for system
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_MAX_QUEUE=512
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_USE_MMAP=1

# Model Loading Optimizations
export OLLAMA_LOAD_TIMEOUT=300  # 5 minutes
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_MAX_VRAM=15360  # 15GB (leave 1GB for system)

# Quantization Preferences (ordered by performance on RTX 5000)
export OLLAMA_PREFERRED_QUANTS="Q4_K_M,Q5_K_M,Q6_K,Q8_0,F16"

# Logging and Monitoring
export OLLAMA_DEBUG=0
export OLLAMA_VERBOSE=0
export CUDA_LAUNCH_BLOCKING=0  # Async execution for performance

# Power Management (for laptop)
export CUDA_POWER_MANAGEMENT=adaptive
export CUDA_THERMAL_THROTTLING=1

# Display GPU Information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    echo ""
fi

# Verify CUDA Installation
if command -v nvcc &> /dev/null; then
    echo "CUDA Compiler Version:"
    nvcc --version | grep "release"
    echo ""
fi

# Performance Tuning Function
rtx5000_tune_performance() {
    echo "Applying RTX 5000 specific performance tuning..."
    
    # Set GPU performance mode (requires nvidia-ml-py)
    python3 -c "
import subprocess
try:
    # Set persistence mode
    subprocess.run(['nvidia-smi', '-pm', '1'], check=True, capture_output=True)
    
    # Set power limit to maximum (if supported)
    subprocess.run(['nvidia-smi', '-pl', '120'], check=True, capture_output=True)
    
    # Set memory and graphics clocks to maximum
    subprocess.run(['nvidia-smi', '-ac', '9001,2115'], check=True, capture_output=True)
    
    print('Performance tuning applied successfully')
except subprocess.CalledProcessError as e:
    print(f'Some performance settings may require root privileges: {e}')
except FileNotFoundError:
    print('nvidia-smi not found, skipping performance tuning')
" 2>/dev/null || echo "Performance tuning requires nvidia-smi and appropriate permissions"
}

# Memory Test Function
rtx5000_memory_test() {
    echo "Testing GPU memory allocation..."
    python3 -c "
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, check=True)
    total, free = map(int, result.stdout.strip().split(', '))
    print(f'Total VRAM: {total} MB')
    print(f'Free VRAM: {free} MB')
    print(f'Recommended max model size: {int(free * 0.8)} MB')
except Exception as e:
    print(f'Could not query GPU memory: {e}')
"
}

# Benchmark Function
rtx5000_benchmark() {
    echo "Running RTX 5000 benchmark..."
    if command -v ollama &> /dev/null; then
        echo "Testing with small model..."
        time ollama run gemma3:1b "Hello, how are you?" --verbose
    else
        echo "Ollama not found, skipping benchmark"
    fi
}

# Main execution
echo "RTX 5000 environment configured!"
echo "Available functions:"
echo "  rtx5000_tune_performance - Apply performance optimizations"
echo "  rtx5000_memory_test      - Test GPU memory"
echo "  rtx5000_benchmark        - Run performance benchmark"
echo ""
echo "To apply performance tuning: rtx5000_tune_performance"
echo "To test memory: rtx5000_memory_test"
echo "To benchmark: rtx5000_benchmark"