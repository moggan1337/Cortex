/**
 * @file memory_bandwidth.hpp
 * @brief Memory Bandwidth Modeling and Analysis
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cortex {
namespace simulation {

/**
 * @brief Memory technology types
 */
enum class MemoryType {
    DRAM,
    DDR4,
    DDR5,
    HBM,
    HBM2,
    HBM3,
    HBM4,
    LPDDR,
    LPDDR5,
    GDDR6,
    SRAM,
    eDRAM
};

/**
 * @brief Memory configuration
 */
struct MemorySpec {
    MemoryType type;
    uint64_t capacity_bytes;
    uint64_t bandwidth_gbps;
    uint32_t bus_width_bits;
    uint32_t burst_size_bytes;
    uint32_t latency_ns;         // CAS latency
    uint32_t tRP_ns;             // Row precharge time
    uint32_t tRCD_ns;            // RAS to CAS delay
    uint32_t tRC_ns;             // Row cycle time
    uint32_t page_size_bytes;
    uint32_t banks;
    uint32_t ranks;
    
    MemorySpec() : type(MemoryType::HBM2), capacity_bytes(8ULL * 1024 * 1024 * 1024),
                   bandwidth_gbps(900), bus_width_bits(4096), burst_size_bytes(64),
                   latency_ns(10), tRP_ns(5), tRCD_ns(5), tRC_ns(40),
                   page_size_bytes(4096), banks(32), ranks(1) {}
};

/**
 * @brief Memory access pattern
 */
struct MemoryAccess {
    uint64_t address;
    uint32_t size_bytes;
    bool is_write;
    uint64_t timestamp_ns;
    uint32_t thread_id;
    
    MemoryAccess() : address(0), size_bytes(0), is_write(false), 
                     timestamp_ns(0), thread_id(0) {}
};

/**
 * @brief Bandwidth measurement result
 */
struct BandwidthMeasurement {
    double peak_bandwidth_gbps;
    double sustained_bandwidth_gbps;
    double average_bandwidth_gbps;
    double efficiency_percent;
    uint64_t total_bytes;
    uint64_t duration_ns;
    uint32_t num_accesses;
    
    BandwidthMeasurement() : peak_bandwidth_gbps(0), sustained_bandwidth_gbps(0),
                              average_bandwidth_gbps(0), efficiency_percent(0),
                              total_bytes(0), duration_ns(0), num_accesses(0) {}
};

/**
 * @brief Memory channel simulation
 */
class MemoryChannel {
public:
    explicit MemoryChannel(const MemorySpec& spec) : spec_(spec) {
        pending_accesses_.reserve(256);
    }
    
    /**
     * @brief Request memory access
     */
    void requestAccess(uint64_t addr, uint32_t size, bool is_write) {
        MemoryAccess access;
        access.address = addr;
        access.size_bytes = size;
        access.is_write = is_write;
        access.timestamp_ns = current_time_ns_;
        pending_accesses_.push_back(access);
    }
    
    /**
     * @brief Advance simulation by one cycle
     */
    void tick(uint64_t cycles = 1) {
        current_time_ns_ += cycles * spec_.latency_ns / 1000;
        processCompletedAccesses();
    }
    
    /**
     * @brief Get current bandwidth usage
     */
    double getCurrentBandwidthGbps() const {
        if (current_time_ns_ == 0) return 0;
        uint64_t bytes_per_period = getBytesTransferred();
        return (bytes_per_period * 8.0) / (current_time_ns_ * 1e-9) / 1e9;
    }
    
    /**
     * @brief Get theoretical peak bandwidth
     */
    double getPeakBandwidthGbps() const { return spec_.bandwidth_gbps; }
    
    /**
     * @brief Process all pending requests
     */
    void drain() {
        while (!pending_accesses_.empty()) {
            tick();
        }
    }
    
    void reset() {
        current_time_ns_ = 0;
        completed_accesses_.clear();
        pending_accesses_.clear();
    }

private:
    void processCompletedAccesses() {
        // Simplified: complete all pending if enough time has passed
        if (current_time_ns_ > 0) {
            for (auto& access : pending_accesses_) {
                completed_accesses_.push_back(access);
            }
            pending_accesses_.clear();
        }
    }
    
    uint64_t getBytesTransferred() const {
        uint64_t total = 0;
        for (const auto& access : completed_accesses_) {
            total += access.size_bytes;
        }
        return total;
    }
    
    MemorySpec spec_;
    uint64_t current_time_ns_ = 0;
    std::vector<MemoryAccess> pending_accesses_;
    std::vector<MemoryAccess> completed_accesses_;
};

/**
 * @brief Memory bandwidth analyzer
 */
class MemoryBandwidthAnalyzer {
public:
    MemoryBandwidthAnalyzer() {
        channels_.reserve(16);
    }
    
    /**
     * @brief Add a memory channel
     */
    void addChannel(const MemorySpec& spec) {
        channels_.emplace_back(spec);
    }
    
    /**
     * @brief Simulate memory access pattern
     */
    BandwidthMeasurement simulateAccessPattern(
        const std::vector<MemoryAccess>& accesses,
        uint64_t duration_ns) {
        
        BandwidthMeasurement result;
        result.duration_ns = duration_ns;
        result.num_accesses = accesses.size();
        
        uint64_t total_bytes = 0;
        double max_bandwidth = 0;
        std::vector<double> bandwidth_samples;
        
        // Simulate access stream
        uint64_t current_time = 0;
        for (const auto& access : accesses) {
            total_bytes += access.size_bytes;
            
            // Calculate transfer time
            double transfer_time_ns = (access.size_bytes * 8.0) / 
                                      (channels_.empty() ? 900e9 : 
                                       channels_[0].getPeakBandwidthGbps()) * 1e9;
            
            current_time += transfer_time_ns;
            
            double instantaneous_bw = (access.size_bytes * 8.0) / 
                                       (transfer_time_ns * 1e-9) / 1e9;
            bandwidth_samples.push_back(instantaneous_bw);
            max_bandwidth = std::max(max_bandwidth, instantaneous_bw);
        }
        
        result.total_bytes = total_bytes;
        result.peak_bandwidth_gbps = max_bandwidth;
        result.average_bandwidth_gbps = (duration_ns > 0) ?
            (total_bytes * 8.0) / (duration_ns * 1e-9) / 1e9 : 0;
        
        // Calculate sustained bandwidth (95th percentile)
        if (!bandwidth_samples.empty()) {
            std::sort(bandwidth_samples.begin(), bandwidth_samples.end());
            size_t idx_95 = (size_t)(bandwidth_samples.size() * 0.95);
            result.sustained_bandwidth_gbps = bandwidth_samples[idx_95];
        }
        
        result.efficiency_percent = channels_.empty() ? 0 :
            (result.average_bandwidth_gbps / channels_[0].getPeakBandwidthGbps()) * 100;
        
        return result;
    }
    
    /**
     * @brief Estimate bandwidth for GEMM operation
     */
    struct GEMMBandwidthEstimate {
        double activation_bandwidth_gbps;
        double weight_bandwidth_gbps;
        double output_bandwidth_gbps;
        double total_bandwidth_gbps;
        double arithmetic_intensity;
        double roofline_performance_tflops;
    };
    
    GEMMBandwidthEstimate estimateGEMMBandwidth(
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t activation_bus_width = 4096,
        uint32_t weight_bus_width = 4096) {
        
        GEMMBandwidthEstimate estimate;
        
        // Activation reads: M * K elements
        uint64_t activation_bytes = (uint64_t)M * K * 2;  // FP16
        // Weight reads: K * N elements
        uint64_t weight_bytes = (uint64_t)K * N * 2;  // FP16
        // Output writes: M * N elements
        uint64_t output_bytes = (uint64_t)M * N * 2;  // FP16
        
        // Estimate time based on bandwidth
        double act_bw = activation_bus_width;  // bits per clock
        double wt_bw = weight_bus_width;
        
        uint64_t total_bytes = activation_bytes + weight_bytes + output_bytes;
        double total_bandwidth_bits_per_clock = act_bw + wt_bw;
        
        estimate.activation_bandwidth_gbps = act_bw / 1e9 * 1000;  // Simplified
        estimate.weight_bandwidth_gbps = wt_bw / 1e9 * 1000;
        estimate.output_bandwidth_gbps = (output_bytes * 8.0 / 1e9) * 1000;
        estimate.total_bandwidth_gbps = estimate.activation_bandwidth_gbps + 
                                        estimate.weight_bandwidth_gbps +
                                        estimate.output_bandwidth_gbps;
        
        // Arithmetic intensity: FLOPs / bytes
        uint64_t flops = 2ULL * M * N * K;
        estimate.arithmetic_intensity = static_cast<double>(flops) / total_bytes;
        
        return estimate;
    }

private:
    std::vector<MemoryChannel> channels_;
};

/**
 * @brief Data movement cost estimator
 */
class DataMovementEstimator {
public:
    struct DataMovementCost {
        uint64_t bytes_transferred;
        uint64_t activation_bytes;
        uint64_t weight_bytes;
        uint64_t output_bytes;
        uint64_t scratchpad_bytes;
        double bandwidth_cost_ns;
        double total_cost_ns;
    };
    
    /**
     * @brief Estimate data movement for a layer
     */
    static DataMovementCost estimateLayerCost(
        const std::vector<std::vector<int>>& input_shape,
        const std::vector<int>& weight_shape,
        const std::vector<int>& output_shape,
        bool use_fusion = false,
        bool in-place = false) {
        
        DataMovementCost cost;
        
        // Calculate bytes
        auto computeBytes = [](const std::vector<std::vector<int>>& shape) -> uint64_t {
            uint64_t total = 1;
            for (const auto& dim : shape) {
                for (int d : dim) {
                    total *= d;
                }
            }
            return total * 2;  // FP16
        };
        
        auto computeBytesSimple = [](const std::vector<int>& shape) -> uint64_t {
            uint64_t total = 1;
            for (int d : shape) {
                total *= d;
            }
            return total * 2;  // FP16
        };
        
        cost.activation_bytes = computeBytesSimple(input_shape);
        cost.weight_bytes = computeBytesSimple(weight_shape);
        cost.output_bytes = computeBytesSimple(output_shape);
        
        if (in_place) {
            cost.output_bytes = 0;  // Reuse activation memory
        }
        
        if (use_fusion) {
            cost.scratchpad_bytes = cost.activation_bytes;  // Fusion buffer
        } else {
            cost.scratchpad_bytes = 0;
        }
        
        cost.bytes_transferred = cost.activation_bytes + 
                                  cost.weight_bytes + 
                                  cost.output_bytes +
                                  cost.scratchpad_bytes;
        
        // Estimate time at 900 Gbps
        double bandwidth_gbps = 900.0;
        cost.bandwidth_cost_ns = (cost.bytes_transferred * 8.0) / 
                                  (bandwidth_gbps * 1e9) * 1e9;
        
        // Compute cost (simplified)
        uint64_t flops = 1;
        for (int d : output_shape) flops *= d;
        flops *= weight_shape[0];  // K dimension
        
        double compute_time_ns = flops / 100e9 * 1e9;  // 100 TFLOPS
        cost.total_cost_ns = cost.bandwidth_cost_ns + compute_time_ns;
        
        return cost;
    }
};

/**
 * @brief Cache hierarchy simulator
 */
class CacheHierarchy {
public:
    struct CacheConfig {
        uint32_t level;          // L1, L2, L3
        uint64_t size_bytes;
        uint32_t line_size_bytes;
        uint32_t associativity;
        uint32_t hit_latency_ns;
        uint32_t bandwidth_gbps;
    };
    
    struct CacheStats {
        uint64_t hits;
        uint64_t misses;
        uint64_t evictions;
        double hit_rate;
        uint64_t total_accesses;
        uint64_t total_bytes_transferred;
    };
    
    explicit CacheHierarchy(const std::vector<CacheConfig>& configs) 
        : configs_(configs) {
        for (const auto& cfg : configs) {
            cache_data_.push_back(std::vector<uint8_t>(cfg.size_bytes, 0));
            stats_.push_back(CacheStats{0, 0, 0, 0, 0, 0});
        }
    }
    
    /**
     * @brief Access memory at address
     */
    bool access(uint64_t address, uint32_t size, bool is_write) {
        bool hit = false;
        
        for (size_t level = 0; level < configs_.size(); ++level) {
            const auto& cfg = configs_[level];
            uint64_t index = (address / cfg.line_size_bytes) % (cfg.size_bytes / cfg.line_size_bytes);
            
            if (index < cfg.size_bytes / cfg.line_size_bytes) {
                hit = true;
                stats_[level].hits++;
                break;
            }
        }
        
        if (!hit && !configs_.empty()) {
            stats_.back().misses++;
        }
        
        updateStats();
        return hit;
    }
    
    const CacheStats& getStats(size_t level) const {
        static CacheStats empty;
        return level < stats_.size() ? stats_[level] : empty;
    }
    
    std::vector<CacheStats> getAllStats() const { return stats_; }
    
    void reset() {
        for (auto& stat : stats_) {
            stat = CacheStats{0, 0, 0, 0, 0, 0};
        }
    }

private:
    void updateStats() {
        for (auto& stat : stats_) {
            stat.total_accesses = stat.hits + stat.misses;
            stat.hit_rate = stat.total_accesses > 0 ?
                (double)stat.hits / stat.total_accesses : 0;
        }
    }
    
    std::vector<CacheConfig> configs_;
    std::vector<std::vector<uint8_t>> cache_data_;
    std::vector<CacheStats> stats_;
};

} // namespace simulation
} // namespace cortex
