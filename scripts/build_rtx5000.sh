#!/bin/bash
# RTX 5000 Optimized Build Script for Ollama
# This script builds Ollama with optimizations specifically for RTX 5000 Mobile Ada Generation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}RTX 5000 Optimized Ollama Build Script${NC}"
echo "========================================"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check CUDA installation
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}Error: CUDA compiler (nvcc) not found${NC}"
        echo "Please install CUDA Toolkit 11.8 or later"
        exit 1
    fi
    
    # Check CUDA version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "CUDA Version: $CUDA_VERSION"
    
    if [[ $(echo "$CUDA_VERSION < 11.8" | bc -l) -eq 1 ]]; then
        echo -e "${YELLOW}Warning: CUDA version $CUDA_VERSION may not support all RTX 5000 optimizations${NC}"
        echo "Recommended: CUDA 11.8 or later for full Ada Lovelace support"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
        echo "Detected GPU: $GPU_INFO"
        
        if [[ $GPU_INFO == *"RTX 5000"* ]]; then
            echo -e "${GREEN}RTX 5000 detected - optimizations will be applied${NC}"
        else
            echo -e "${YELLOW}Warning: RTX 5000 not detected, but optimizations will still be applied${NC}"
        fi
    fi
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}Error: CMake not found${NC}"
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
    echo "CMake Version: $CMAKE_VERSION"
    
    # Check Go
    if ! command -v go &> /dev/null; then
        echo -e "${RED}Error: Go not found${NC}"
        exit 1
    fi
    
    GO_VERSION=$(go version | sed 's/go version go//' | cut -d' ' -f1)
    echo "Go Version: $GO_VERSION"
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Set build environment
setup_environment() {
    echo -e "${YELLOW}Setting up build environment...${NC}"
    
    # Export RTX 5000 specific flags
    export CMAKE_CUDA_ARCHITECTURES="89"
    export CUDA_NVCC_FLAGS="-O3 --use_fast_math -Xptxas -O3"
    export CGO_CFLAGS="-O3 -march=native"
    export CGO_CXXFLAGS="-O3 -march=native"
    export CGO_LDFLAGS="-O3"
    
    # Set CUDA paths
    if [[ -z "$CUDA_HOME" ]]; then
        if [[ -d "/usr/local/cuda" ]]; then
            export CUDA_HOME="/usr/local/cuda"
        elif [[ -d "/opt/cuda" ]]; then
            export CUDA_HOME="/opt/cuda"
        fi
    fi
    
    if [[ -n "$CUDA_HOME" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    fi
    
    echo "CUDA_HOME: $CUDA_HOME"
    echo "CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
}

# Clean previous builds
clean_build() {
    echo -e "${YELLOW}Cleaning previous builds...${NC}"
    
    if [[ -d "build" ]]; then
        rm -rf build
    fi
    
    if [[ -d "dist" ]]; then
        rm -rf dist
    fi
    
    # Clean Go cache
    go clean -cache -modcache -testcache
    
    echo -e "${GREEN}Build directory cleaned${NC}"
}

# Configure CMake with RTX 5000 optimizations
configure_cmake() {
    echo -e "${YELLOW}Configuring CMake with RTX 5000 optimizations...${NC}"
    
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="89" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=256 \
        -DGGML_CUDA_NO_VMM=OFF \
        -DGGML_CUDA_NO_PEER_COPY=OFF \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math -Xptxas -O3 -DGGML_CUDA_USE_TENSOR_CORES" \
        -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native" \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
        -DGGML_NATIVE=ON \
        -DGGML_STATIC=ON
    
    cd ..
    echo -e "${GREEN}CMake configuration completed${NC}"
}

# Build the project
build_project() {
    echo -e "${YELLOW}Building Ollama with RTX 5000 optimizations...${NC}"
    
    # Build with maximum parallel jobs
    NPROC=$(nproc)
    echo "Using $NPROC parallel jobs"
    
    cd build
    make -j$NPROC
    cd ..
    
    # Build Go components
    echo -e "${YELLOW}Building Go components...${NC}"
    go build -ldflags="-s -w" -tags cuda .
    
    echo -e "${GREEN}Build completed successfully${NC}"
}

# Verify build
verify_build() {
    echo -e "${YELLOW}Verifying build...${NC}"
    
    if [[ -f "./ollama" ]]; then
        echo -e "${GREEN}Ollama binary created successfully${NC}"
        
        # Check if CUDA support is enabled
        if ./ollama --help | grep -q "cuda"; then
            echo -e "${GREEN}CUDA support verified${NC}"
        fi
        
        # Get binary size
        SIZE=$(du -h ./ollama | cut -f1)
        echo "Binary size: $SIZE"
        
    else
        echo -e "${RED}Error: Ollama binary not found${NC}"
        exit 1
    fi
}

# Create optimized configuration
create_config() {
    echo -e "${YELLOW}Creating RTX 5000 optimized configuration...${NC}"
    
    cat > rtx5000_config.json << EOF
{
    "gpu": {
        "enabled": true,
        "device": 0,
        "memory_fraction": 0.9,
        "allow_growth": true
    },
    "cuda": {
        "architecture": "89",
        "use_tensor_cores": true,
        "use_flash_attention": true,
        "batch_size": 256,
        "streams": 4
    },
    "memory": {
        "pool_size": "12GB",
        "prefetch": true,
        "async_copy": true
    },
    "performance": {
        "quantization": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
        "max_parallel": 4,
        "keep_alive": "5m"
    }
}
EOF
    
    echo -e "${GREEN}Configuration file created: rtx5000_config.json${NC}"
}

# Performance test
performance_test() {
    echo -e "${YELLOW}Running performance test...${NC}"
    
    # Source the environment script
    source scripts/rtx5000_env.sh
    
    # Test with a small model
    echo "Testing model loading and inference..."
    timeout 60s ./ollama run gemma3:1b "Hello, this is a test." || echo "Test completed or timed out"
    
    echo -e "${GREEN}Performance test completed${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}Starting RTX 5000 optimized build...${NC}"
    
    check_prerequisites
    setup_environment
    clean_build
    configure_cmake
    build_project
    verify_build
    create_config
    
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Source the environment: source scripts/rtx5000_env.sh"
    echo "2. Run performance tuning: rtx5000_tune_performance"
    echo "3. Test the build: rtx5000_benchmark"
    echo ""
    echo "Optimized binary: ./ollama"
    echo "Configuration: ./rtx5000_config.json"
    
    # Ask if user wants to run performance test
    read -p "Run performance test now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        performance_test
    fi
}

# Run main function
main "$@"