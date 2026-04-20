/**
 * @file systolic_array.hpp
 * @brief Systolic Array Architecture for Matrix Multiplication
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <optional>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace cortex {
namespace core {

/**
 * @brief Data types supported by the systolic array
 */
enum class DataType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4
};

/**
 * @brief Precision configuration for computations
 */
struct PrecisionConfig {
    DataType dtype;
    bool use_mixed_precision;
    bool use_tf32;
    
    PrecisionConfig(DataType dt = DataType::FP16) 
        : dtype(dt), use_mixed_precision(false), use_tf32(false) {}
};

/**
 * @brief Systolic array configuration
 */
struct SystolicArrayConfig {
    uint32_t rows;           // Number of processing elements in rows
    uint32_t cols;           // Number of processing elements in columns
    uint32_t clock_freq_hz;  // Operating frequency in Hz
    uint32_t vreg_size;      // Vector register size
    uint32_t accum_stages;   // Accumulation pipeline stages
    
    SystolicArrayConfig(uint32_t r = 128, uint32_t c = 128, uint32_t freq = 1000000000)
        : rows(r), cols(c), clock_freq_hz(freq), vreg_size(512), accum_stages(4) {}
    
    uint32_t totalPes() const { return rows * cols; }
    double peakTflops() const { 
        return (static_cast<double>(rows * cols) * clock_freq_hz * 2) / 1e12; 
    }
};

/**
 * @brief Result of a systolic array operation
 */
struct SystolicArrayResult {
    uint64_t cycles;
    uint64_t bytes_read;
    uint64_t bytes_written;
    double latency_us;
    double throughput_tflops;
    double arithmetic_intensity;
    
    SystolicArrayResult() 
        : cycles(0), bytes_read(0), bytes_written(0), latency_us(0), 
          throughput_tflops(0), arithmetic_intensity(0) {}
};

/**
 * @brief Processing Element (PE) - Fundamental unit of systolic array
 */
class ProcessingElement {
public:
    ProcessingElement() : acc(0.0f), input_val(0.0f), weight_val(0.0f) {}
    
    /**
     * @brief Perform one step of MAC operation
     * @param input Input activation value
     * @param weight Weight value
     * @return Result of accumulation
     */
    float step(float input, float weight) {
        acc += input * weight;
        input_val = input;
        weight_val = weight;
        return acc;
    }
    
    /**
     * @brief Reset the PE for new computation
     */
    void reset() { acc = 0.0f; input_val = 0.0f; weight_val = 0.0f; }
    
    /**
     * @brief Get accumulated value
     */
    float getAccumulator() const { return acc; }
    
    /**
     * @brief Set accumulator value
     */
    void setAccumulator(float val) { acc = val; }

private:
    float acc;        // Accumulator
    float input_val;  // Input register
    float weight_val; // Weight register
};

/**
     * @brief Main systolic array class for matrix multiplication
     */
template<typename T = float>
class SystolicArray {
public:
    using Matrix = std::vector<std::vector<T>>;
    
    explicit SystolicArray(const SystolicArrayConfig& config)
        : config_(config), 
          pes_(config.rows, std::vector<ProcessingElement>(config.cols)),
          pipeline_stages_(config.accum_stages, 0.0f) {
        reset();
    }
    
    /**
     * @brief Perform matrix multiplication using weight-stationary dataflow
     * @param activation Input activation matrix [M][K]
     * @param weight Weight matrix [K][N]
     * @param precision Precision configuration
     * @return Operation result with timing information
     */
    SystolicArrayResult multiply(const Matrix& activation, 
                                 const Matrix& weight,
                                 const PrecisionConfig& precision = PrecisionConfig()) {
        SystolicArrayResult result;
        
        uint32_t M = activation.size();
        uint32_t K = activation.empty() ? 0 : activation[0].size();
        uint32_t N = weight.empty() ? 0 : weight[0].size();
        
        if (M == 0 || K == 0 || N == 0) {
            return result;
        }
        
        // Calculate operation metrics
        uint64_t flops = 2ULL * M * K * N;  // MACs count as 2 FLOPs
        uint64_t bytes_accessed = (M * K + K * N + M * N) * sizeof(T);
        
        // Estimate cycles based on dataflow
        uint32_t array_size = std::min(config_.rows, config_.cols);
        uint32_t tiles_m = (M + array_size - 1) / array_size;
        uint32_t tiles_n = (N + array_size - 1) / array_size;
        uint32_t tiles_k = K / array_size;
        
        // Pipeline depth accounts for systolic array latency
        uint32_t pipeline_depth = array_size + config_.accum_stages;
        
        uint64_t cycles_per_tile = 2 * array_size + pipeline_depth;
        uint64_t total_cycles = tiles_m * tiles_n * tiles_k * cycles_per_tile;
        
        // Adjust for different data types
        double dtype_multiplier = getDtypeMultiplier(precision.dtype);
        
        result.cycles = total_cycles;
        result.bytes_read = bytes_accessed;
        result.bytes_written = M * N * sizeof(T);
        result.arithmetic_intensity = static_cast<double>(flops) / bytes_accessed;
        result.latency_us = (total_cycles * dtype_multiplier) / (config_.clock_freq_hz / 1e6);
        result.throughput_tflops = (flops * dtype_multiplier) / (result.latency_us * 1e-6) / 1e12;
        
        return result;
    }
    
    /**
     * @brief Perform batch matrix multiplication
     */
    std::vector<SystolicArrayResult> batchMultiply(
        const std::vector<Matrix>& activations,
        const Matrix& weights,
        const PrecisionConfig& precision = PrecisionConfig()) {
        
        std::vector<SystolicArrayResult> results;
        results.reserve(activations.size());
        
        for (const auto& activation : activations) {
            results.push_back(multiply(activation, weights, precision));
        }
        
        return results;
    }
    
    /**
     * @brief Get configuration
     */
    const SystolicArrayConfig& getConfig() const { return config_; }
    
    /**
     * @brief Reset all processing elements
     */
    void reset() {
        for (auto& row : pes_) {
            for (auto& pe : row) {
                pe.reset();
            }
        }
        std::fill(pipeline_stages_.begin(), pipeline_stages_.end(), 0.0f);
    }
    
    /**
     * @brief Simulate cycle-by-cycle execution
     */
    void simulateCycle(uint32_t cycle, const Matrix& activation, 
                       const Matrix& weight, Matrix& output,
                       uint32_t m_offset, uint32_t n_offset);
    
private:
    double getDtypeMultiplier(DataType dtype) const {
        switch (dtype) {
            case DataType::FP32: return 1.0;
            case DataType::FP16: return 0.5;
            case DataType::BF16: return 0.5;
            case DataType::INT8: return 0.25;
            case DataType::INT4: return 0.125;
            default: return 1.0;
        }
    }
    
    SystolicArrayConfig config_;
    std::vector<std::vector<ProcessingElement>> pes_;
    std::vector<float> pipeline_stages_;
};

/**
 * @brief Windowed embedding lookup accelerator
 */
class EmbeddingAccelerator {
public:
    EmbeddingAccelerator(uint32_t vocab_size, uint32_t embedding_dim, 
                          uint32_t max_sequence_length)
        : vocab_size_(vocab_size),
          embedding_dim_(embedding_dim),
          max_sequence_length_(max_sequence_length) {}
    
    struct EmbeddingResult {
        uint64_t cycles;
        uint64_t memory_accesses;
        double bandwidth_gbps;
        std::vector<float> output;
    };
    
    EmbeddingResult lookup(const std::vector<uint32_t>& indices,
                          const std::vector<float>& embedding_table) {
        EmbeddingResult result;
        
        uint32_t num_tokens = indices.size();
        uint32_t lookups = num_tokens;
        
        // Memory access: each lookup accesses embedding_dim values
        result.memory_accesses = static_cast<uint64_t>(lookups) * embedding_dim_;
        
        // Estimate cycles based on memory bandwidth
        double bytes_per_access = embedding_dim_ * sizeof(float);
        double total_bytes = result.memory_accesses * sizeof(float);
        double estimated_bandwidth_gbps = 1000.0;  // Assumed memory bandwidth
        result.bandwidth_gbps = estimated_bandwidth_gbps;
        result.cycles = static_cast<uint64_t>(total_bytes / (estimated_bandwidth_gbps / 8.0));
        
        // Compute output
        result.output.resize(num_tokens * embedding_dim_, 0.0f);
        for (uint32_t t = 0; t < num_tokens; ++t) {
            uint32_t idx = indices[t];
            if (idx < vocab_size_) {
                uint32_t offset = idx * embedding_dim_;
                for (uint32_t d = 0; d < embedding_dim_; ++d) {
                    result.output[t * embedding_dim_ + d] = 
                        embedding_table[offset + d];
                }
            }
        }
        
        return result;
    }

private:
    uint32_t vocab_size_;
    uint32_t embedding_dim_;
    uint32_t max_sequence_length_;
};

} // namespace core
} // namespace cortex
