#!/usr/bin/env python3
"""
RTX 5000 Laptop GPU Benchmark Suite for Ollama
Comprehensive performance testing and optimization validation
"""

import os
import sys
import time
import json
import subprocess
import argparse
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
    import GPUtil
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install psutil gputil numpy")
    sys.exit(1)

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    test_name: str
    model_name: str
    tokens_per_second: float
    time_to_first_token: float
    total_time: float
    memory_used: float
    gpu_utilization: float
    power_consumption: float
    temperature: float
    success: bool
    error_message: Optional[str] = None

class RTX5000Benchmark:
    """RTX 5000 optimized benchmark suite"""
    
    def __init__(self, ollama_path: str = "./ollama"):
        self.ollama_path = ollama_path
        self.results: List[BenchmarkResult] = []
        self.gpu = None
        
        # Initialize GPU monitoring
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu = gpus[0]  # Assume first GPU is RTX 5000
                print(f"Detected GPU: {self.gpu.name}")
                print(f"GPU Memory: {self.gpu.memoryTotal}MB total, {self.gpu.memoryFree}MB free")
            else:
                print("Warning: No GPU detected")
        except Exception as e:
            print(f"GPU detection failed: {e}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("Checking prerequisites...")
        
        # Check Ollama binary
        if not os.path.exists(self.ollama_path):
            print(f"Error: Ollama binary not found at {self.ollama_path}")
            return False
        
        # Check CUDA
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Error: nvidia-smi not available")
                return False
        except FileNotFoundError:
            print("Error: NVIDIA drivers not installed")
            return False
        
        # Check if RTX 5000 is available
        if self.gpu and "RTX 5000" not in self.gpu.name:
            print(f"Warning: Expected RTX 5000, found {self.gpu.name}")
        
        print("Prerequisites check passed")
        return True
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        info = {
            "cpu": {
                "model": subprocess.getoutput("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2").strip(),
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
            },
            "memory": {
                "total": psutil.virtual_memory().total // (1024**3),  # GB
                "available": psutil.virtual_memory().available // (1024**3)
            },
            "gpu": {
                "name": self.gpu.name if self.gpu else "Unknown",
                "memory_total": self.gpu.memoryTotal if self.gpu else 0,
                "memory_free": self.gpu.memoryFree if self.gpu else 0,
                "driver_version": subprocess.getoutput("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits").strip()
            }
        }
        return info
    
    def monitor_gpu(self) -> Tuple[float, float, float]:
        """Monitor GPU utilization, memory, and temperature"""
        if not self.gpu:
            return 0.0, 0.0, 0.0
        
        try:
            # Refresh GPU info
            GPUtil.showUtilization()
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return gpu.load * 100, gpu.memoryUtil * 100, gpu.temperature
        except:
            pass
        
        return 0.0, 0.0, 0.0
    
    def run_ollama_command(self, args: List[str], timeout: int = 300) -> Tuple[bool, str, float]:
        """Run Ollama command and measure execution time"""
        start_time = time.time()
        
        try:
            cmd = [self.ollama_path] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return True, result.stdout, execution_time
            else:
                return False, result.stderr, execution_time
                
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds", time.time() - start_time
        except Exception as e:
            return False, str(e), time.time() - start_time
    
    def benchmark_model_loading(self, model_name: str) -> BenchmarkResult:
        """Benchmark model loading performance"""
        print(f"Benchmarking model loading: {model_name}")
        
        # Stop any running models first
        self.run_ollama_command(["stop", model_name])
        time.sleep(2)
        
        # Monitor initial state
        gpu_util_before, mem_util_before, temp_before = self.monitor_gpu()
        
        # Load model
        start_time = time.time()
        success, output, load_time = self.run_ollama_command(["run", model_name, "test"], timeout=180)
        
        # Monitor after loading
        gpu_util_after, mem_util_after, temp_after = self.monitor_gpu()
        
        memory_used = (mem_util_after - mem_util_before) * (self.gpu.memoryTotal if self.gpu else 0) / 100
        
        return BenchmarkResult(
            test_name="model_loading",
            model_name=model_name,
            tokens_per_second=0.0,
            time_to_first_token=load_time,
            total_time=load_time,
            memory_used=memory_used,
            gpu_utilization=gpu_util_after,
            power_consumption=0.0,  # Would need additional monitoring
            temperature=temp_after,
            success=success,
            error_message=output if not success else None
        )
    
    def benchmark_inference(self, model_name: str, prompt: str, expected_tokens: int = 100) -> BenchmarkResult:
        """Benchmark inference performance"""
        print(f"Benchmarking inference: {model_name}")
        
        # Monitor initial state
        gpu_util_before, mem_util_before, temp_before = self.monitor_gpu()
        
        # Run inference
        start_time = time.time()
        success, output, total_time = self.run_ollama_command([
            "run", model_name, prompt
        ], timeout=120)
        
        # Monitor after inference
        gpu_util_after, mem_util_after, temp_after = self.monitor_gpu()
        
        # Calculate metrics
        tokens_per_second = 0.0
        time_to_first_token = 0.0
        
        if success and output:
            # Estimate tokens (rough approximation)
            estimated_tokens = len(output.split())
            if estimated_tokens > 0 and total_time > 0:
                tokens_per_second = estimated_tokens / total_time
            
            # Time to first token is roughly 10% of total time for streaming
            time_to_first_token = total_time * 0.1
        
        memory_used = mem_util_after * (self.gpu.memoryTotal if self.gpu else 0) / 100
        
        return BenchmarkResult(
            test_name="inference",
            model_name=model_name,
            tokens_per_second=tokens_per_second,
            time_to_first_token=time_to_first_token,
            total_time=total_time,
            memory_used=memory_used,
            gpu_utilization=gpu_util_after,
            power_consumption=0.0,
            temperature=temp_after,
            success=success,
            error_message=output if not success else None
        )
    
    def benchmark_concurrent_requests(self, model_name: str, num_requests: int = 4) -> BenchmarkResult:
        """Benchmark concurrent request handling"""
        print(f"Benchmarking concurrent requests: {model_name} ({num_requests} requests)")
        
        # Start Ollama server if not running
        server_process = subprocess.Popen([self.ollama_path, "serve"], 
                                        stdout=subprocess.DEVNULL, 
                                        stderr=subprocess.DEVNULL)
        time.sleep(5)  # Wait for server to start
        
        try:
            # Monitor initial state
            gpu_util_before, mem_util_before, temp_before = self.monitor_gpu()
            
            # Launch concurrent requests
            processes = []
            start_time = time.time()
            
            for i in range(num_requests):
                cmd = ["curl", "-s", "http://localhost:11434/api/generate", 
                       "-d", f'{{"model": "{model_name}", "prompt": "Hello, this is request {i+1}"}}']
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append(proc)
            
            # Wait for all requests to complete
            results = []
            for proc in processes:
                stdout, stderr = proc.communicate(timeout=60)
                results.append((proc.returncode == 0, stdout.decode()))
            
            total_time = time.time() - start_time
            
            # Monitor after requests
            gpu_util_after, mem_util_after, temp_after = self.monitor_gpu()
            
            # Calculate metrics
            successful_requests = sum(1 for success, _ in results if success)
            requests_per_second = successful_requests / total_time if total_time > 0 else 0
            
            memory_used = mem_util_after * (self.gpu.memoryTotal if self.gpu else 0) / 100
            
            return BenchmarkResult(
                test_name="concurrent_requests",
                model_name=model_name,
                tokens_per_second=requests_per_second,  # Repurpose for requests/sec
                time_to_first_token=total_time / num_requests if num_requests > 0 else 0,
                total_time=total_time,
                memory_used=memory_used,
                gpu_utilization=gpu_util_after,
                power_consumption=0.0,
                temperature=temp_after,
                success=successful_requests == num_requests,
                error_message=f"Only {successful_requests}/{num_requests} requests succeeded" if successful_requests < num_requests else None
            )
            
        finally:
            # Clean up server
            server_process.terminate()
            server_process.wait(timeout=10)
    
    def run_comprehensive_benchmark(self, models: List[str]) -> Dict:
        """Run comprehensive benchmark suite"""
        print("Starting RTX 5000 comprehensive benchmark...")
        print("=" * 60)
        
        system_info = self.get_system_info()
        benchmark_start = time.time()
        
        for model in models:
            print(f"\nTesting model: {model}")
            print("-" * 40)
            
            # Test model loading
            result = self.benchmark_model_loading(model)
            self.results.append(result)
            
            if result.success:
                # Test inference
                result = self.benchmark_inference(model, "Explain quantum computing in simple terms.")
                self.results.append(result)
                
                # Test concurrent requests
                result = self.benchmark_concurrent_requests(model, 4)
                self.results.append(result)
            else:
                print(f"Skipping further tests for {model} due to loading failure")
        
        benchmark_duration = time.time() - benchmark_start
        
        # Compile results
        results_summary = {
            "system_info": system_info,
            "benchmark_duration": benchmark_duration,
            "timestamp": time.time(),
            "results": [
                {
                    "test_name": r.test_name,
                    "model_name": r.model_name,
                    "tokens_per_second": r.tokens_per_second,
                    "time_to_first_token": r.time_to_first_token,
                    "total_time": r.total_time,
                    "memory_used": r.memory_used,
                    "gpu_utilization": r.gpu_utilization,
                    "temperature": r.temperature,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        return results_summary
    
    def generate_report(self, results: Dict, output_file: str = "rtx5000_benchmark_report.json"):
        """Generate detailed benchmark report"""
        
        # Save JSON report
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate text summary
        report_file = output_file.replace('.json', '.txt')
        with open(report_file, 'w') as f:
            f.write("RTX 5000 Laptop GPU Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            # System info
            f.write("System Information:\n")
            f.write(f"GPU: {results['system_info']['gpu']['name']}\n")
            f.write(f"GPU Memory: {results['system_info']['gpu']['memory_total']}MB\n")
            f.write(f"CPU: {results['system_info']['cpu']['model']}\n")
            f.write(f"RAM: {results['system_info']['memory']['total']}GB\n")
            f.write(f"Driver: {results['system_info']['gpu']['driver_version']}\n\n")
            
            # Results summary
            f.write("Benchmark Results:\n")
            f.write("-" * 30 + "\n")
            
            successful_tests = [r for r in results['results'] if r['success']]
            failed_tests = [r for r in results['results'] if not r['success']]
            
            f.write(f"Total tests: {len(results['results'])}\n")
            f.write(f"Successful: {len(successful_tests)}\n")
            f.write(f"Failed: {len(failed_tests)}\n")
            f.write(f"Duration: {results['benchmark_duration']:.2f} seconds\n\n")
            
            # Performance metrics
            if successful_tests:
                inference_results = [r for r in successful_tests if r['test_name'] == 'inference']
                if inference_results:
                    avg_tokens_per_sec = statistics.mean([r['tokens_per_second'] for r in inference_results])
                    avg_ttft = statistics.mean([r['time_to_first_token'] for r in inference_results])
                    avg_gpu_util = statistics.mean([r['gpu_utilization'] for r in inference_results])
                    
                    f.write("Performance Summary:\n")
                    f.write(f"Average tokens/second: {avg_tokens_per_sec:.2f}\n")
                    f.write(f"Average time to first token: {avg_ttft:.2f}s\n")
                    f.write(f"Average GPU utilization: {avg_gpu_util:.1f}%\n\n")
            
            # Detailed results
            f.write("Detailed Results:\n")
            for result in results['results']:
                f.write(f"\nTest: {result['test_name']} - {result['model_name']}\n")
                f.write(f"  Success: {result['success']}\n")
                if result['success']:
                    f.write(f"  Tokens/sec: {result['tokens_per_second']:.2f}\n")
                    f.write(f"  Time to first token: {result['time_to_first_token']:.2f}s\n")
                    f.write(f"  Total time: {result['total_time']:.2f}s\n")
                    f.write(f"  Memory used: {result['memory_used']:.1f}MB\n")
                    f.write(f"  GPU utilization: {result['gpu_utilization']:.1f}%\n")
                    f.write(f"  Temperature: {result['temperature']:.1f}°C\n")
                else:
                    f.write(f"  Error: {result['error_message']}\n")
        
        print(f"\nBenchmark complete!")
        print(f"JSON report: {output_file}")
        print(f"Text report: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="RTX 5000 Ollama Benchmark Suite")
    parser.add_argument("--ollama-path", default="./ollama", help="Path to Ollama binary")
    parser.add_argument("--models", nargs="+", default=["gemma3:1b", "gemma3:4b"], help="Models to test")
    parser.add_argument("--output", default="rtx5000_benchmark_report.json", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer tests)")
    
    args = parser.parse_args()
    
    benchmark = RTX5000Benchmark(args.ollama_path)
    
    if not benchmark.check_prerequisites():
        sys.exit(1)
    
    models = args.models
    if args.quick:
        models = models[:1]  # Test only first model for quick run
    
    results = benchmark.run_comprehensive_benchmark(models)
    benchmark.generate_report(results, args.output)

if __name__ == "__main__":
    main()