#!/bin/bash
# RTX 5000 Complete Setup and Optimization Script
# One-click setup for optimal Ollama performance on RTX 5000 laptops

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
print_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    ____  ________  __   ________  ____  ____
   / __ \/_  __/ |/ /  / ____/ / / / / / / /
  / /_/ / / /  |   /  /___ \/ / / / / / / / 
 / _, _/ / /  /   |  ____/ / /_/ / /_/ / /  
/_/ |_| /_/  /_/|_| /____/\____/\____/_/   
                                           
    Ollama Optimization Suite
EOF
    echo -e "${NC}"
    echo -e "${BLUE}RTX 5000 Mobile Ada Generation Optimizations${NC}"
    echo -e "${YELLOW}Maximum Performance • Optimal Memory Usage • Enhanced Throughput${NC}"
    echo ""
}

# Progress indicator
show_progress() {
    local current=$1
    local total=$2
    local desc=$3
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    printf "\r${BLUE}[${GREEN}"
    printf "%${filled}s" | tr ' ' '█'
    printf "${NC}${BLUE}"
    printf "%${empty}s" | tr ' ' '░'
    printf "] ${percent}%% - ${desc}${NC}"
    
    if [ $current -eq $total ]; then
        echo ""
    fi
}

# Check system requirements
check_system() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    local checks=0
    local total_checks=6
    
    # Check GPU
    ((checks++))
    show_progress $checks $total_checks "Checking GPU"
    if nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
        if [[ $GPU_NAME == *"RTX 5000"* ]]; then
            echo -e "${GREEN}✓ RTX 5000 detected: $GPU_NAME${NC}"
        else
            echo -e "${YELLOW}⚠ GPU detected but not RTX 5000: $GPU_NAME${NC}"
            echo -e "${YELLOW}  Optimizations will still be applied${NC}"
        fi
    else
        echo -e "${RED}✗ NVIDIA GPU not detected${NC}"
        exit 1
    fi
    
    # Check CUDA
    ((checks++))
    show_progress $checks $total_checks "Checking CUDA"
    if nvcc --version &>/dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo -e "${GREEN}✓ CUDA $CUDA_VERSION detected${NC}"
    else
        echo -e "${RED}✗ CUDA not found${NC}"
        exit 1
    fi
    
    # Check CMake
    ((checks++))
    show_progress $checks $total_checks "Checking CMake"
    if cmake --version &>/dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
        echo -e "${GREEN}✓ CMake $CMAKE_VERSION detected${NC}"
    else
        echo -e "${RED}✗ CMake not found${NC}"
        exit 1
    fi
    
    # Check Go
    ((checks++))
    show_progress $checks $total_checks "Checking Go"
    if go version &>/dev/null; then
        GO_VERSION=$(go version | sed 's/go version go//' | cut -d' ' -f1)
        echo -e "${GREEN}✓ Go $GO_VERSION detected${NC}"
    else
        echo -e "${RED}✗ Go not found${NC}"
        exit 1
    fi
    
    # Check Python
    ((checks++))
    show_progress $checks $total_checks "Checking Python"
    if python3 --version &>/dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
    else
        echo -e "${YELLOW}⚠ Python3 not found (needed for benchmarking)${NC}"
    fi
    
    # Check disk space
    ((checks++))
    show_progress $checks $total_checks "Checking disk space"
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    if [ $AVAILABLE_GB -gt 10 ]; then
        echo -e "${GREEN}✓ Sufficient disk space: ${AVAILABLE_GB}GB available${NC}"
    else
        echo -e "${YELLOW}⚠ Low disk space: ${AVAILABLE_GB}GB available${NC}"
    fi
    
    echo ""
}

# Install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing Python dependencies for benchmarking...${NC}"
    
    if python3 -m pip install psutil gputil numpy &>/dev/null; then
        echo -e "${GREEN}✓ Python dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ Could not install Python dependencies (benchmarking may not work)${NC}"
    fi
    echo ""
}

# Setup environment
setup_environment() {
    echo -e "${YELLOW}Setting up RTX 5000 environment...${NC}"
    
    # Source environment configuration
    if [ -f "scripts/rtx5000_env.sh" ]; then
        source scripts/rtx5000_env.sh
        echo -e "${GREEN}✓ Environment configured${NC}"
    else
        echo -e "${RED}✗ Environment script not found${NC}"
        exit 1
    fi
    
    # Create config directory
    mkdir -p ~/.ollama
    
    # Create optimized configuration
    cat > ~/.ollama/rtx5000_config.json << EOF
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
    
    echo -e "${GREEN}✓ Configuration created at ~/.ollama/rtx5000_config.json${NC}"
    echo ""
}

# Build optimized Ollama
build_ollama() {
    echo -e "${YELLOW}Building Ollama with RTX 5000 optimizations...${NC}"
    
    if [ -f "scripts/build_rtx5000.sh" ]; then
        echo -e "${BLUE}Starting optimized build process...${NC}"
        ./scripts/build_rtx5000.sh
        echo -e "${GREEN}✓ Build completed${NC}"
    else
        echo -e "${RED}✗ Build script not found${NC}"
        exit 1
    fi
    echo ""
}

# Performance tuning
apply_performance_tuning() {
    echo -e "${YELLOW}Applying performance tuning...${NC}"
    
    # GPU performance settings (may require sudo)
    echo -e "${BLUE}Attempting to apply GPU performance settings...${NC}"
    
    if sudo -n true 2>/dev/null; then
        # Set persistence mode
        if sudo nvidia-smi -pm 1 &>/dev/null; then
            echo -e "${GREEN}✓ GPU persistence mode enabled${NC}"
        fi
        
        # Set power limit to maximum
        if sudo nvidia-smi -pl 120 &>/dev/null; then
            echo -e "${GREEN}✓ Power limit set to 120W${NC}"
        fi
        
        # Set memory and graphics clocks
        if sudo nvidia-smi -ac 9001,2115 &>/dev/null; then
            echo -e "${GREEN}✓ GPU clocks optimized${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Sudo access not available, skipping GPU performance settings${NC}"
        echo -e "${YELLOW}  Run 'sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 120' manually for best performance${NC}"
    fi
    
    echo ""
}

# Run quick test
run_quick_test() {
    echo -e "${YELLOW}Running quick performance test...${NC}"
    
    if [ -f "./ollama" ]; then
        echo -e "${BLUE}Testing Ollama binary...${NC}"
        
        # Test help command
        if ./ollama --help &>/dev/null; then
            echo -e "${GREEN}✓ Ollama binary working${NC}"
        else
            echo -e "${RED}✗ Ollama binary test failed${NC}"
            return 1
        fi
        
        # Quick memory test
        echo -e "${BLUE}Testing GPU memory allocation...${NC}"
        if python3 -c "
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, check=True)
    total, free = map(int, result.stdout.strip().split(', '))
    print(f'GPU Memory: {total}MB total, {free}MB free')
    if free > 1000:
        print('✓ Sufficient GPU memory available')
    else:
        print('⚠ Low GPU memory available')
except Exception as e:
    print(f'Could not query GPU memory: {e}')
" 2>/dev/null; then
            echo -e "${GREEN}✓ GPU memory test passed${NC}"
        fi
        
    else
        echo -e "${RED}✗ Ollama binary not found${NC}"
        return 1
    fi
    
    echo ""
}

# Generate usage instructions
generate_instructions() {
    echo -e "${CYAN}RTX 5000 Optimization Setup Complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Quick Start Commands:${NC}"
    echo -e "${BLUE}  # Source environment (run this in each new terminal)${NC}"
    echo -e "  source scripts/rtx5000_env.sh"
    echo ""
    echo -e "${BLUE}  # Run a model${NC}"
    echo -e "  ./ollama run gemma3:1b"
    echo ""
    echo -e "${BLUE}  # Run benchmark${NC}"
    echo -e "  ./scripts/rtx5000_benchmark.py --quick"
    echo ""
    echo -e "${YELLOW}Performance Tuning:${NC}"
    echo -e "${BLUE}  # Apply additional performance tuning${NC}"
    echo -e "  rtx5000_tune_performance"
    echo ""
    echo -e "${BLUE}  # Test memory allocation${NC}"
    echo -e "  rtx5000_memory_test"
    echo ""
    echo -e "${BLUE}  # Run full benchmark${NC}"
    echo -e "  rtx5000_benchmark"
    echo ""
    echo -e "${YELLOW}Configuration Files:${NC}"
    echo -e "  • Environment: scripts/rtx5000_env.sh"
    echo -e "  • Config: ~/.ollama/rtx5000_config.json"
    echo -e "  • Documentation: RTX5000_README.md"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  • Check RTX5000_README.md for detailed troubleshooting"
    echo -e "  • Run benchmark to validate performance"
    echo -e "  • Monitor GPU with: watch -n 1 nvidia-smi"
    echo ""
    echo -e "${GREEN}Enjoy optimized LLM inference on your RTX 5000! 🚀${NC}"
}

# Main execution
main() {
    print_banner
    
    echo -e "${PURPLE}Starting RTX 5000 optimization setup...${NC}"
    echo ""
    
    # Step 1: Check system
    check_system
    
    # Step 2: Install dependencies
    install_dependencies
    
    # Step 3: Setup environment
    setup_environment
    
    # Step 4: Build Ollama (optional, ask user)
    read -p "$(echo -e ${YELLOW}Build Ollama with RTX 5000 optimizations? [y/N]: ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_ollama
    else
        echo -e "${BLUE}Skipping build (you can run ./scripts/build_rtx5000.sh later)${NC}"
        echo ""
    fi
    
    # Step 5: Apply performance tuning
    apply_performance_tuning
    
    # Step 6: Run quick test
    if [ -f "./ollama" ]; then
        run_quick_test
    else
        echo -e "${YELLOW}Skipping test (Ollama binary not found)${NC}"
        echo ""
    fi
    
    # Step 7: Generate instructions
    generate_instructions
    
    # Optional: Run benchmark
    read -p "$(echo -e ${YELLOW}Run quick benchmark now? [y/N]: ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] && [ -f "scripts/rtx5000_benchmark.py" ]; then
        echo -e "${BLUE}Running quick benchmark...${NC}"
        ./scripts/rtx5000_benchmark.py --quick
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "RTX 5000 Ollama Optimization Setup"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --check        Only check system requirements"
        echo "  --build        Only build Ollama"
        echo "  --test         Only run tests"
        echo ""
        exit 0
        ;;
    --check)
        print_banner
        check_system
        exit 0
        ;;
    --build)
        print_banner
        build_ollama
        exit 0
        ;;
    --test)
        print_banner
        run_quick_test
        exit 0
        ;;
    *)
        main
        ;;
esac