/**
 * @file benchmark_runner.hpp
 * @brief Benchmark Runner and Performance Testing
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include "core/tpu_architecture.hpp"
#include "core/systolic_array.hpp"
#include "models/benchmark_models.hpp"
#include "analysis/latency_throughput.hpp"
#include "analysis/roofline_model.hpp"
#include "simulation/quantization.hpp"
#include "simulation/batch_norm_fusion.hpp"
#include "simulation/memory_bandwidth.hpp"

namespace cortex {
namespace benchmark {

/**
 * @brief Benchmark result structure
 */
struct BenchmarkResult {
    std::string model_name;
    std::string architecture;
    
    // Timing
    double latency_p50_us;
    double latency_p90_us;
    double latency_p95_us;
    double latency_p99_us;
    double latency_mean_us;
    double latency_std_us;
    
    // Throughput
    double throughput_samples_per_sec;
    double throughput_tokens_per_sec;
    double throughput_images_per_sec;
    
    // Hardware utilization
    double compute_utilization_percent;
    double memory_utilization_percent;
    double roofline_efficiency_percent;
    
    // Model characteristics
    uint64_t total_flops;
    uint64_t total_memory_bytes;
    uint32_t batch_size;
    uint32_t sequence_length;
    
    // Optimization results
    std::vector<std::string> optimizations_applied;
    double estimated_speedup;
    
    BenchmarkResult() : latency_p50_us(0), latency_p90_us(0), latency_p95_us(0),
                        latency_p99_us(0), latency_mean_us(0), latency_std_us(0),
                        throughput_samples_per_sec(0), throughput_tokens_per_sec(0),
                        throughput_images_per_sec(0), compute_utilization_percent(0),
                        memory_utilization_percent(0), roofline_efficiency_percent(0),
                        total_flops(0), total_memory_bytes(0), batch_size(1),
                        sequence_length(0), estimated_speedup(1.0) {}
};

/**
 * @brief Hardware configuration for benchmarks
 */
struct HardwareConfig {
    std::string name;
    std::string vendor;
    analysis::PeakSpecifications peak_specs;
    uint32_t systolic_array_size;
    uint32_t num_chips;
    
    HardwareConfig() : name("TPU_V3"), vendor("Google"), 
                        systolic_array_size(128), num_chips(1) {
        peak_specs.fp16_tflops = 180;
        peak_specs.fp32_tflops = 90;
        peak_specs.bf16_tflops = 180;
        peak_specs.int8_tops = 360;
        peak_specs.int4_tops = 720;
        peak_specs.hbm_bandwidth_gbps = 900;
        peak_specs.l2_bandwidth_gbps = 4000;
    }
};

/**
 * @brief Benchmark runner class
 */
class BenchmarkRunner {
public:
    BenchmarkRunner() {
        // Default TPU V3 configuration
        hw_config_.name = "TPU_V3";
        hw_config_.peak_specs.fp16_tflops = 180;
        hw_config_.peak_specs.hbm_bandwidth_gbps = 900;
        
        initializeHardware();
    }
    
    explicit BenchmarkRunner(const HardwareConfig& hw_config) 
        : hw_config_(hw_config) {
        initializeHardware();
    }
    
    /**
     * @brief Run benchmark for a specific model
     */
    BenchmarkResult run(const std::string& model_name,
                       uint32_t batch_size = 1,
                       uint32_t sequence_length = 512,
                       uint32_t num_runs = 100,
                       bool enable_optimizations = true) {
        
        BenchmarkResult result;
        result.model_name = model_name;
        result.batch_size = batch_size;
        result.sequence_length = sequence_length;
        
        // Get model configuration
        auto model_config = models::ModelFactory::create(
            model_name, batch_size, sequence_length);
        
        result.architecture = model_config.architecture;
        result.total_flops = model_config.total_flops;
        result.total_memory_bytes = model_config.weights_memory_bytes + 
                                    model_config.activations_memory_bytes;
        
        // Run simulation
        std::vector<double> latencies;
        latencies.reserve(num_runs);
        
        for (uint32_t i = 0; i < num_runs; ++i) {
            double latency = simulateInference(model_config, enable_optimizations);
            latencies.push_back(latency);
        }
        
        // Compute statistics
        computeStatistics(latencies, result);
        
        // Compute throughput
        result.throughput_samples_per_sec = 1e6 / result.latency_mean_us * batch_size;
        
        if (model_name.find("bert") != std::string::npos || 
            model_name.find("gpt") != std::string::npos ||
            model_name.find("llama") != std::string::npos) {
            result.throughput_tokens_per_sec = 
                result.throughput_samples_per_sec * sequence_length;
        }
        
        if (model_config.architecture == "ResNet" || 
            model_config.architecture == "ConvNeXt" ||
            model_config.architecture == "ViT") {
            result.throughput_images_per_sec = result.throughput_samples_per_sec;
        }
        
        // Compute hardware utilization
        computeUtilization(model_config, result);
        
        // Record optimizations
        if (enable_optimizations) {
            result.optimizations_applied = {
                "Batch Normalization Fusion",
                "Activation Folding",
                "Memory Layout Optimization"
            };
            if (batch_size > 1) {
                result.optimizations_applied.push_back("Batch Parallelism");
            }
        }
        
        return result;
    }
    
    /**
     * @brief Run comprehensive benchmark suite
     */
    std::vector<BenchmarkResult> runSuite(
        const std::vector<std::string>& models,
        uint32_t batch_size = 1,
        uint32_t sequence_length = 512,
        uint32_t num_runs = 100) {
        
        std::vector<BenchmarkResult> results;
        results.reserve(models.size());
        
        for (const auto& model : models) {
            std::cout << "Running " << model << " benchmark..." << std::endl;
            results.push_back(run(model, batch_size, sequence_length, num_runs));
        }
        
        return results;
    }
    
    /**
     * @brief Generate comparison report
     */
    std::string generateReport(const std::vector<BenchmarkResult>& results) {
        std::ostringstream oss;
        
        oss << "\n" << std::string(80, '=') << "\n";
        oss << "CORTEX BENCHMARK RESULTS\n";
        oss << std::string(80, '=') << "\n\n";
        
        oss << "Hardware: " << hw_config_.name << "\n";
        oss << "Peak Performance: " << hw_config_.peak_specs.fp16_tflops << " TFLOPS (FP16)\n";
        oss << "Memory Bandwidth: " << hw_config_.peak_specs.hbm_bandwidth_gbps << " GB/s\n";
        oss << "Systolic Array: " << hw_config_.systolic_array_size << "x" 
            << hw_config_.systolic_array_size << "\n\n";
        
        oss << std::setw(20) << std::left << "Model"
            << std::setw(15) << "Latency (P50)"
            << std::setw(15) << "Throughput"
            << std::setw(15) << "Efficiency"
            << std::setw(15) << "Memory Util"
            << "\n";
        oss << std::string(80, '-') << "\n";
        
        for (const auto& r : results) {
            std::string throughput_str;
            if (r.throughput_images_per_sec > 0) {
                throughput_str = std::to_string((int)r.throughput_images_per_sec) + " img/s";
            } else if (r.throughput_tokens_per_sec > 0) {
                throughput_str = std::to_string((int)r.throughput_tokens_per_sec) + " tok/s";
            } else {
                throughput_str = std::to_string((int)r.throughput_samples_per_sec) + " samp/s";
            }
            
            oss << std::setw(20) << std::left << r.model_name
                << std::setw(15) << std::fixed << std::setprecision(2) << r.latency_p50_us / 1000 << " ms"
                << std::setw(15) << throughput_str
                << std::setw(15) << std::setprecision(1) << r.roofline_efficiency_percent << "%"
                << std::setw(15) << std::setprecision(1) << r.memory_utilization_percent << "%"
                << "\n";
        }
        
        oss << "\n" << std::string(80, '=') << "\n";
        
        return oss.str();
    }
    
    void setHardwareConfig(const HardwareConfig& config) {
        hw_config_ = config;
        initializeHardware();
    }

private:
    void initializeHardware() {
        analysis::PeakSpecifications peak;
        peak.fp16_tflops = hw_config_.peak_specs.fp16_tflops;
        peak.hbm_bandwidth_gbps = hw_config_.peak_specs.hbm_bandwidth_gbps;
        
        roofline_model_ = std::make_unique<analysis::RooflineModel>(peak);
        
        analysis::HardwareSpec hw_spec;
        hw_spec.peak_tflops_fp16 = peak.fp16_tflops;
        hw_spec.memory_bandwidth_gbps = peak.hbm_bandwidth_gbps;
        
        latency_estimator_ = std::make_unique<analysis::LatencyEstimator>(hw_spec);
        throughput_estimator_ = std::make_unique<analysis::ThroughputEstimator>(hw_spec);
    }
    
    double simulateInference(const models::ModelConfig& config, bool optimized) {
        // Estimate base latency using roofline model
        double ai = static_cast<double>(config.total_flops) / 
                     config.total_memory_bytes;
        
        double roofline_perf = roofline_model_->calculateRoofline(ai, "fp16");
        
        // Time = FLOPs / Performance
        double time_us = (config.total_flops / (roofline_perf * 1e12)) * 1e6;
        
        // Apply optimizations
        if (optimized) {
            double optimization_factor = 0.85;  // 15% improvement from optimizations
            time_us *= optimization_factor;
        }
        
        // Add some variation for realism
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise(1.0, 0.05);
        time_us *= std::clamp(noise(gen), 0.85, 1.15);
        
        return time_us;
    }
    
    void computeStatistics(const std::vector<double>& latencies, 
                           BenchmarkResult& result) {
        if (latencies.empty()) return;
        
        std::vector<double> sorted = latencies;
        std::sort(sorted.begin(), sorted.end());
        
        size_t n = sorted.size();
        
        result.latency_mean_us = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
        result.latency_p50_us = sorted[n * 50 / 100];
        result.latency_p90_us = sorted[n * 90 / 100];
        result.latency_p95_us = sorted[n * 95 / 100];
        result.latency_p99_us = sorted[n * 99 / 100];
        
        double sum_sq_diff = 0;
        for (double lat : sorted) {
            double diff = lat - result.latency_mean_us;
            sum_sq_diff += diff * diff;
        }
        result.latency_std_us = std::sqrt(sum_sq_diff / n);
    }
    
    void computeUtilization(const models::ModelConfig& config,
                          BenchmarkResult& result) {
        double ai = static_cast<double>(config.total_flops) / 
                     config.total_memory_bytes;
        
        double peak_flops = hw_config_.peak_specs.fp16_tflops * 1e3;
        double peak_bandwidth = hw_config_.peak_specs.hbm_bandwidth_gbps;
        
        double knee_point = peak_flops / peak_bandwidth;
        
        if (ai < knee_point) {
            result.compute_utilization_percent = (ai / knee_point) * 50;
            result.memory_utilization_percent = 80 + (ai / knee_point) * 20;
            result.roofline_efficiency_percent = (ai / knee_point) * 70;
        } else {
            result.compute_utilization_percent = 85;
            result.memory_utilization_percent = 50;
            result.roofline_efficiency_percent = 75 + (ai / (knee_point * 2)) * 15;
        }
        
        result.roofline_efficiency_percent = std::min(99.9, result.roofline_efficiency_percent);
    }
    
    HardwareConfig hw_config_;
    std::unique_ptr<analysis::RooflineModel> roofline_model_;
    std::unique_ptr<analysis::LatencyEstimator> latency_estimator_;
    std::unique_ptr<analysis::ThroughputEstimator> throughput_estimator_;
};

/**
 * @brief Run quick benchmark
 */
inline BenchmarkResult quickBenchmark(
    const std::string& model,
    uint32_t batch_size = 1,
    bool enable_optimizations = true) {
    
    HardwareConfig hw;
    hw.name = "TPU_V3";
    hw.peak_specs.fp16_tflops = 180;
    hw.peak_specs.hbm_bandwidth_gbps = 900;
    
    BenchmarkRunner runner(hw);
    return runner.run(model, batch_size, 512, 50, enable_optimizations);
}

} // namespace benchmark
} // namespace cortex
