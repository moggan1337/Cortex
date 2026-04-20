/**
 * @file cortex.hpp
 * @brief Cortex - Neural Network Hardware Accelerator Emulator
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include "core/systolic_array.hpp"
#include "core/tpu_architecture.hpp"
#include "simulation/memory_bandwidth.hpp"
#include "simulation/dataflow_engine.hpp"
#include "simulation/batch_norm_fusion.hpp"
#include "simulation/quantization.hpp"
#include "analysis/latency_throughput.hpp"
#include "analysis/roofline_model.hpp"

namespace cortex {

/**
 * @brief Version information
 */
constexpr const char* VERSION = "1.0.0";
constexpr const char* VERSION_MAJOR = "1";
constexpr const char* VERSION_MINOR = "0";
constexpr const char* VERSION_PATCH = "0";

/**
 * @brief Configuration for the emulator
 */
struct EmulatorConfig {
    // Hardware configuration
    uint32_t systolic_array_size;
    uint32_t num_tpu_cores;
    uint32_t clock_frequency_hz;
    uint64_t memory_capacity_gb;
    uint32_t memory_bandwidth_gbps;
    
    // Simulation configuration
    bool enable_roofline_analysis;
    bool enable_quantization;
    bool enable_batch_norm_fusion;
    bool enable_dataflow_optimization;
    uint32_t num_simulation_runs;
    
    // Precision configuration
    core::DataType default_precision;
    bool enable_mixed_precision;
    
    EmulatorConfig() 
        : systolic_array_size(128),
          num_tpu_cores(1),
          clock_frequency_hz(1000000000),
          memory_capacity_gb(8),
          memory_bandwidth_gbps(900),
          enable_roofline_analysis(true),
          enable_quantization(true),
          enable_batch_norm_fusion(true),
          enable_dataflow_optimization(true),
          num_simulation_runs(100),
          default_precision(core::DataType::FP16),
          enable_mixed_precision(true) {}
};

/**
 * @brief Main emulator class
 */
class CortexEmulator {
public:
    explicit CortexEmulator(const EmulatorConfig& config = EmulatorConfig())
        : config_(config) {
        initialize();
    }
    
    /**
     * @brief Run comprehensive benchmark
     */
    struct BenchmarkResult {
        std::string model_name;
        double total_latency_us;
        double throughput_samples_per_sec;
        double peak_tflops;
        double achieved_tflops;
        double roofline_efficiency;
        double memory_bandwidth_utilization;
        std::vector<std::string> optimizations_applied;
        analysis::LatencyEstimate latency_stats;
    };
    
    BenchmarkResult runBenchmark(const std::string& model_config);

private:
    void initialize() {
        // Initialize TPU architecture
        tpu_ = std::make_unique<core::TPUArchitecture>(
            core::TPUGeneration::TPU_V3);
        
        // Initialize analyzers
        analysis::PeakSpecifications peak_spec;
        peak_spec.fp16_tflops = 180;
        peak_spec.hbm_bandwidth_gbps = config_.memory_bandwidth_gbps;
        
        roofline_ = std::make_unique<analysis::RooflineModel>(peak_spec);
        
        analysis::HardwareSpec hw_spec;
        hw_spec.peak_tflops_fp16 = peak_spec.fp16_tflops;
        hw_spec.memory_bandwidth_gbps = config_.memory_bandwidth_gbps;
        hw_spec.num_cores = config_.num_tpu_cores;
        hw_spec.frequency_mhz = config_.clock_frequency_hz / 1000000;
        
        latency_estimator_ = std::make_unique<analysis::LatencyEstimator>(hw_spec);
        throughput_estimator_ = std::make_unique<analysis::ThroughputEstimator>(hw_spec);
        profiler_ = std::make_unique<analysis::PerformanceProfiler>(hw_spec);
        
        // Initialize dataflow engine
        simulation::ExecutionConfig exec_config;
        dataflow_engine_ = std::make_unique<simulation::DataflowEngine>(exec_config);
        
        // Initialize batch norm fusion
        bn_fusion_ = std::make_unique<simulation::BatchNormFusion>();
        
        // Initialize quantization
        quant_calibrator_ = std::make_unique<simulation::QuantizationCalibrator>();
    }
    
    EmulatorConfig config_;
    std::unique_ptr<core::TPUArchitecture> tpu_;
    std::unique_ptr<analysis::RooflineModel> roofline_;
    std::unique_ptr<analysis::LatencyEstimator> latency_estimator_;
    std::unique_ptr<analysis::ThroughputEstimator> throughput_estimator_;
    std::unique_ptr<analysis::PerformanceProfiler> profiler_;
    std::unique_ptr<simulation::DataflowEngine> dataflow_engine_;
    std::unique_ptr<simulation::BatchNormFusion> bn_fusion_;
    std::unique_ptr<simulation::QuantizationCalibrator> quant_calibrator_;
};

/**
 * @brief Print version information
 */
inline void printVersion() {
    std::cout << "Cortex v" << VERSION << std::endl;
    std::cout << "Neural Network Hardware Accelerator Emulator" << std::endl;
}

/**
 * @brief Print hardware information
 */
inline void printHardwareInfo(const EmulatorConfig& config) {
    std::cout << "=== Hardware Configuration ===" << std::endl;
    std::cout << "Systolic Array Size: " << config.systolic_array_size << "x" 
              << config.systolic_array_size << std::endl;
    std::cout << "TPU Cores: " << config.num_tpu_cores << std::endl;
    std::cout << "Clock Frequency: " << config.clock_frequency_hz / 1e9 << " GHz" << std::endl;
    std::cout << "Memory Capacity: " << config.memory_capacity_gb << " GB" << std::endl;
    std::cout << "Memory Bandwidth: " << config.memory_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "Default Precision: ";
    switch (config.default_precision) {
        case core::DataType::FP32: std::cout << "FP32"; break;
        case core::DataType::FP16: std::cout << "FP16"; break;
        case core::DataType::BF16: std::cout << "BF16"; break;
        case core::DataType::INT8: std::cout << "INT8"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;
}

} // namespace cortex

#include <iostream>

#endif // CORTEX_HPP_
