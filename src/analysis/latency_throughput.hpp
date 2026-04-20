/**
 * @file latency_throughput.hpp
 * @brief Latency and Throughput Estimation for Neural Networks
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

namespace cortex {
namespace analysis {

/**
 * @brief Hardware specification
 */
struct HardwareSpec {
    double peak_tflops_fp16;
    double peak_tflops_fp32;
    double peak_tflops_int8;
    uint32_t memory_bandwidth_gbps;
    uint32_t l2_cache_mb;
    uint32_t scratchpad_kb;
    uint32_t num_cores;
    uint32_t frequency_mhz;
    
    HardwareSpec() : peak_tflops_fp16(0), peak_tflops_fp32(0), peak_tflops_int8(0),
                     memory_bandwidth_gbps(900), l2_cache_mb(32), 
                     scratchpad_kb(128), num_cores(1), frequency_mhz(1000) {}
};

/**
 * @brief Operation statistics
 */
struct OpStats {
    std::string name;
    std::string type;
    uint64_t flops;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t compute_cycles;
    uint64_t memory_cycles;
    uint32_t tile_count;
    
    OpStats() : flops(0), bytes_read(0), bytes_written(0),
                compute_cycles(0), memory_cycles(0), tile_count(1) {}
    
    uint64_t totalCycles() const { return std::max(compute_cycles, memory_cycles); }
    double arithmeticIntensity() const {
        uint64_t total_bytes = bytes_read + bytes_written;
        return total_bytes > 0 ? static_cast<double>(flops) / total_bytes : 0;
    }
};

/**
 * @brief Latency result
 */
struct LatencyEstimate {
    double p50_us;
    double p90_us;
    double p95_us;
    double p99_us;
    double mean_us;
    double std_us;
    double min_us;
    double max_us;
    
    LatencyEstimate() : p50_us(0), p90_us(0), p95_us(0), p99_us(0),
                        mean_us(0), std_us(0), min_us(0), max_us(0) {}
};

/**
 * @brief Throughput result
 */
struct ThroughputEstimate {
    double samples_per_second;
    double tokens_per_second;
    double images_per_second;
    double inference_per_second;
    uint32_t batch_size;
    double latency_ms;
    double gpu_utilization_percent;
    
    ThroughputEstimate() : samples_per_second(0), tokens_per_second(0),
                           images_per_second(0), inference_per_second(0),
                           batch_size(1), latency_ms(0), gpu_utilization_percent(0) {}
};

/**
 * @brief Latency estimator
 */
class LatencyEstimator {
public:
    explicit LatencyEstimator(const HardwareSpec& hw) : hw_(hw) {}
    
    /**
     * @brief Estimate latency for a single operation
     */
    double estimateOpLatency(const OpStats& op, const std::string& dtype = "fp16") {
        double flops_factor = 1.0;
        if (dtype == "fp32") flops_factor = 1.0;
        else if (dtype == "fp16") flops_factor = 2.0;
        else if (dtype == "int8") flops_factor = 4.0;
        
        double peak_flops = hw_.peak_tflops_fp16 * 1e12 * flops_factor;
        
        // Compute-bound time
        double compute_time_us = (op.flops / peak_flops) * 1e6;
        
        // Memory-bound time
        double memory_bandwidth_bytes = hw_.memory_bandwidth_gbps * 1e9 / 8.0;
        double memory_time_us = (op.bytes_read + op.bytes_written) / 
                                 memory_bandwidth_bytes * 1e6;
        
        // Take max (either compute or memory bound)
        double time_us = std::max(compute_time_us, memory_time_us);
        
        // Adjust for tiling overhead
        if (op.tile_count > 1) {
            time_us *= (1.0 + 0.05 * std::log2(op.tile_count));
        }
        
        return time_us;
    }
    
    /**
     * @brief Estimate latency for a sequence of operations
     */
    LatencyEstimate estimateSequenceLatency(
        const std::vector<OpStats>& ops,
        bool parallel = false) {
        
        LatencyEstimate result;
        
        if (ops.empty()) return result;
        
        std::vector<double> op_latencies;
        op_latencies.reserve(ops.size());
        
        for (const auto& op : ops) {
            op_latencies.push_back(estimateOpLatency(op));
        }
        
        if (parallel) {
            // Pipeline parallel: use max of concurrent operations
            // Simplified: just use sum for now
            double total_latency = std::accumulate(op_latencies.begin(), 
                                                   op_latencies.end(), 0.0);
            result.mean_us = total_latency;
            result.p50_us = total_latency;
            result.p90_us = total_latency;
        } else {
            // Sequential: sum of all latencies
            double total_latency = std::accumulate(op_latencies.begin(), 
                                                   op_latencies.end(), 0.0);
            
            // Compute statistics
            result.mean_us = total_latency / ops.size();
            
            double sum_sq_diff = 0;
            for (double lat : op_latencies) {
                double diff = lat - result.mean_us;
                sum_sq_diff += diff * diff;
            }
            result.std_us = std::sqrt(sum_sq_diff / ops.size());
            
            // Percentiles
            std::vector<double> sorted = op_latencies;
            std::sort(sorted.begin(), sorted.end());
            
            size_t n = sorted.size();
            result.min_us = sorted.front();
            result.max_us = sorted.back();
            result.p50_us = sorted[n * 50 / 100];
            result.p90_us = sorted[n * 90 / 100];
            result.p95_us = sorted[n * 95 / 100];
            result.p99_us = sorted[n * 99 / 100];
        }
        
        return result;
    }
    
    /**
     * @brief Monte Carlo simulation for latency distribution
     */
    LatencyEstimate monteCarloEstimate(
        const std::vector<OpStats>& ops,
        uint32_t num_samples = 1000) {
        
        LatencyEstimate result;
        std::vector<double> samples;
        samples.reserve(num_samples);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (uint32_t i = 0; i < num_samples; ++i) {
            double total_latency = 0;
            
            for (const auto& op : ops) {
                // Add random variation
                std::normal_distribution<> noise(1.0, 0.1);
                double factor = std::clamp(noise(gen), 0.7, 1.3);
                total_latency += estimateOpLatency(op) * factor;
            }
            
            samples.push_back(total_latency);
        }
        
        std::sort(samples.begin(), samples.end());
        
        size_t n = samples.size();
        result.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
        result.min_us = samples.front();
        result.max_us = samples.back();
        result.p50_us = samples[n * 50 / 100];
        result.p90_us = samples[n * 90 / 100];
        result.p95_us = samples[n * 95 / 100];
        result.p99_us = samples[n * 99 / 100];
        
        double sum_sq_diff = 0;
        for (double s : samples) {
            double diff = s - result.mean_us;
            sum_sq_diff += diff * diff;
        }
        result.std_us = std::sqrt(sum_sq_diff / n);
        
        return result;
    }

private:
    HardwareSpec hw_;
};

/**
 * @brief Throughput estimator
 */
class ThroughputEstimator {
public:
    explicit ThroughputEstimator(const HardwareSpec& hw) : hw_(hw) {}
    
    /**
     * @brief Estimate throughput for a model
     */
    ThroughputEstimate estimateThroughput(
        uint64_t total_flops,
        uint64_t total_memory_bytes,
        uint32_t batch_size = 1) {
        
        ThroughputEstimate result;
        result.batch_size = batch_size;
        
        // Peak compute time
        double peak_flops = hw_.peak_tflops_fp16 * 1e12;
        double compute_time_s = total_flops / peak_flops;
        
        // Memory time
        double memory_bandwidth = hw_.memory_bandwidth_gbps * 1e9 / 8.0;
        double memory_time_s = total_memory_bytes / memory_bandwidth;
        
        // Total time per sample (accounting for batch)
        double time_per_sample = std::max(compute_time_s, memory_time_s);
        
        // Latency
        result.latency_ms = time_per_sample * 1000;
        
        // Throughput
        result.samples_per_second = 1.0 / time_per_sample * batch_size;
        
        // GPU utilization estimate
        double bound_time = std::max(compute_time_s, memory_time_s);
        result.gpu_utilization_percent = 
            (compute_time_s / bound_time) * 50 + 30;  // Simplified model
        
        return result;
    }
    
    /**
     * @brief Estimate optimal batch size
     */
    uint32_t findOptimalBatchSize(
        uint64_t total_flops,
        uint64_t total_memory_bytes,
        uint64_t available_memory_bytes) {
        
        // Simple model: memory scales linearly with batch
        uint64_t memory_per_sample = total_memory_bytes;
        uint32_t max_batch = available_memory_bytes / memory_per_sample;
        
        // Find sweet spot (typically 8-64 for most accelerators)
        uint32_t optimal = 1;
        double best_efficiency = 0;
        
        for (uint32_t b = 1; b <= max_batch && b <= 256; b *= 2) {
            auto throughput = estimateThroughput(total_flops, 
                                                  total_memory_bytes * b, b);
            
            // Efficiency is samples/second per batch size
            double efficiency = throughput.samples_per_second / b;
            
            if (efficiency > best_efficiency) {
                best_efficiency = efficiency;
                optimal = b;
            }
        }
        
        return optimal;
    }
    
    /**
     * @brief Estimate pipeline throughput
     */
    ThroughputEstimate estimatePipelineThroughput(
        const std::vector<OpStats>& stage_ops,
        uint32_t num_stages,
        uint32_t batch_size = 1) {
        
        ThroughputEstimate result;
        result.batch_size = batch_size;
        
        // Calculate bottleneck stage latency
        double max_latency = 0;
        for (const auto& stage : stage_ops) {
            double stage_latency = 0;
            for (const auto& op : stage_ops) {
                LatencyEstimator le(hw_);
                stage_latency += le.estimateOpLatency(op);
            }
            max_latency = std::max(max_latency, stage_latency);
        }
        
        // Pipeline throughput is limited by slowest stage
        double stage_time_s = max_latency / 1e6;
        result.samples_per_second = batch_size / (stage_time_s * num_stages);
        
        // Latency through entire pipeline
        result.latency_ms = stage_time_s * 1000 * num_stages;
        
        return result;
    }

private:
    HardwareSpec hw_;
};

/**
 * @brief Performance profiler
 */
class PerformanceProfiler {
public:
    PerformanceProfiler(const HardwareSpec& hw) : hw_(hw) {}
    
    /**
     * @brief Profile a neural network layer by layer
     */
    struct ProfileResult {
        std::string model_name;
        uint64_t total_flops;
        uint64_t total_memory_bytes;
        LatencyEstimate end_to_end_latency;
        std::vector<OpStats> layer_stats;
        std::vector<double> layer_latencies_us;
        double compute_bound_percent;
        double memory_bound_percent;
        double roofline_efficiency;
    };
    
    ProfileResult profile(
        const std::string& model_name,
        const std::vector<OpStats>& layer_ops,
        bool detailed = false) {
        
        ProfileResult result;
        result.model_name = model_name;
        
        LatencyEstimator le(hw_);
        ThroughputEstimator te(hw_);
        
        for (size_t i = 0; i < layer_ops.size(); ++i) {
            const auto& op = layer_ops[i];
            result.layer_stats.push_back(op);
            result.total_flops += op.flops;
            result.total_memory_bytes += op.bytes_read + op.bytes_written;
            
            double latency = le.estimateOpLatency(op);
            result.layer_latencies_us.push_back(latency);
        }
        
        // End-to-end latency
        result.end_to_end_latency = le.estimateSequenceLatency(layer_ops);
        
        // Bound analysis
        double total_compute_time = 0;
        double total_memory_time = 0;
        
        for (const auto& op : layer_ops) {
            double peak_flops = hw_.peak_tflops_fp16 * 1e12;
            total_compute_time += op.flops / peak_flops;
            
            double memory_bandwidth = hw_.memory_bandwidth_gbps * 1e9 / 8.0;
            total_memory_time += (op.bytes_read + op.bytes_written) / memory_bandwidth;
        }
        
        double total_time = total_compute_time + total_memory_time;
        result.compute_bound_percent = (total_compute_time / total_time) * 100;
        result.memory_bound_percent = (total_memory_time / total_time) * 100;
        
        // Roofline efficiency
        double ai = static_cast<double>(result.total_flops) / result.total_memory_bytes;
        double peak_flops = hw_.peak_tflops_fp16 * 1e12;
        double peak_bandwidth = hw_.memory_bandwidth_gbps * 1e9 / 8.0;
        double roofline_perf = std::min(peak_flops, ai * peak_bandwidth);
        
        result.roofline_efficiency = 
            (result.end_to_end_latency.mean_us * 1e-6 * peak_flops / 
             result.total_flops) * 100;
        
        return result;
    }
    
    /**
     * @brief Generate performance report
     */
    std::string generateReport(const ProfileResult& result) {
        std::ostringstream oss;
        
        oss << "=== Performance Profile Report ===\n";
        oss << "Model: " << result.model_name << "\n\n";
        
        oss << "Total FLOPs: " << result.total_flops << "\n";
        oss << "Total Memory: " << result.total_memory_bytes / (1024*1024) << " MB\n";
        oss << "End-to-End Latency: " << result.end_to_end_latency.mean_us << " us\n";
        oss << "  P50: " << result.end_to_end_latency.p50_us << " us\n";
        oss << "  P90: " << result.end_to_end_latency.p90_us << " us\n";
        oss << "  P99: " << result.end_to_end_latency.p99_us << " us\n\n";
        
        oss << "Bound Analysis:\n";
        oss << "  Compute Bound: " << result.compute_bound_percent << "%\n";
        oss << "  Memory Bound: " << result.memory_bound_percent << "%\n";
        oss << "  Roofline Efficiency: " << result.roofline_efficiency << "%\n\n";
        
        oss << "Layer Breakdown:\n";
        for (size_t i = 0; i < result.layer_stats.size(); ++i) {
            oss << "  [" << i << "] " << result.layer_stats[i].name 
                << ": " << result.layer_latencies_us[i] << " us\n";
        }
        
        return oss.str();
    }

private:
    HardwareSpec hw_;
};

} // namespace analysis
} // namespace cortex
