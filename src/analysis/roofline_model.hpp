/**
 * @file roofline_model.hpp
 * @brief Roofline Performance Model Analysis
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>

namespace cortex {
namespace analysis {

/**
 * @brief Hardware peak specifications
 */
struct PeakSpecifications {
    double fp32_tflops;
    double fp16_tflops;
    double bf16_tflops;
    double int8_tops;
    double int4_tops;
    
    uint64_t hbm_bandwidth_gbps;
    uint64_t l2_bandwidth_gbps;
    uint64_t scratchpad_bandwidth_gbps;
    
    uint64_t l2_cache_mb;
    uint64_t scratchpad_kb;
    
    PeakSpecifications() 
        : fp32_tflops(0), fp16_tflops(0), bf16_tflops(0), 
          int8_tops(0), int4_tops(0),
          hbm_bandwidth_gbps(900), l2_bandwidth_gbps(4000),
          scratchpad_bandwidth_gbps(16000),
          l2_cache_mb(32), scratchpad_kb(128) {}
};

/**
 * @brief Kernel characteristics
 */
struct KernelCharacteristics {
    std::string name;
    double arithmetic_intensity;      // FLOPs/byte
    double achieved_tflops;
    double achieved_bandwidth_gbps;
    double roofline_bound_tflops;
    std::string bottleneck;
    
    KernelCharacteristics() : arithmetic_intensity(0), achieved_tflops(0),
                              achieved_bandwidth_gbps(0), roofline_bound_tflops(0),
                              bottleneck("unknown") {}
};

/**
 * @brief Roofline model analyzer
 */
class RooflineModel {
public:
    explicit RooflineModel(const PeakSpecifications& peak) : peak_(peak) {}
    
    /**
     * @brief Calculate roofline performance bound
     * @param ai Arithmetic intensity (FLOPs/byte)
     * @param precision Compute precision
     * @return Peak achievable performance in TFLOPS
     */
    double calculateRoofline(double ai, const std::string& precision = "fp16") {
        double peak_flops = getPeakFlops(precision);
        double peak_bandwidth = static_cast<double>(peak_.hbm_bandwidth_gbps);
        
        // Memory bound region
        double memory_bound = ai * peak_bandwidth;
        
        // Compute bound region (ceiling at peak)
        double compute_bound = peak_flops;
        
        return std::min(memory_bound, compute_bound);
    }
    
    /**
     * @brief Analyze a kernel against roofline
     */
    KernelCharacteristics analyzeKernel(
        const std::string& name,
        uint64_t flops,
        uint64_t bytes_accessed,
        const std::string& precision = "fp16") {
        
        KernelCharacteristics result;
        result.name = name;
        
        // Calculate arithmetic intensity
        result.arithmetic_intensity = bytes_accessed > 0 ? 
            static_cast<double>(flops) / bytes_accessed : 0;
        
        // Calculate achieved performance
        double peak_flops = getPeakFlops(precision);
        double peak_bandwidth = static_cast<double>(peak_.hbm_bandwidth_gbps);
        
        // Estimate achieved based on arithmetic intensity
        double roofline_peak = calculateRoofline(result.arithmetic_intensity, precision);
        result.roofline_bound_tflops = roofline_peak;
        
        // Achieved is typically below roofline due to inefficiencies
        result.achieved_tflops = roofline_peak * 0.7;  // 70% of roofline
        
        result.achieved_bandwidth_gbps = 
            result.arithmetic_intensity > 0 ?
            result.achieved_tflops / result.arithmetic_intensity : 0;
        
        // Determine bottleneck
        if (result.arithmetic_intensity < peak_flops / peak_bandwidth) {
            result.bottleneck = "memory";
        } else {
            result.bottleneck = "compute";
        }
        
        return result;
    }
    
    /**
     * @brief Calculate efficiency percentage
     */
    double calculateEfficiency(double achieved_tflops, const std::string& precision = "fp16") {
        double peak = getPeakFlops(precision);
        return peak > 0 ? (achieved_tflops / peak) * 100 : 0;
    }
    
    /**
     * @brief Estimate potential improvement
     */
    double estimateImprovementPotential(const KernelCharacteristics& kernel) {
        double current_perf = kernel.achieved_tflops;
        double roofline_perf = kernel.roofline_bound_tflops;
        
        // Potential is the gap between achieved and roofline
        return (roofline_perf - current_perf) / current_perf * 100;
    }
    
    /**
     * @brief Generate roofline points for plotting
     */
    struct RooflinePoint {
        double arithmetic_intensity;
        double performance;
        std::string region;
    };
    
    std::vector<RooflinePoint> generateRooflineCurve(
        const std::string& precision = "fp16",
        uint32_t num_points = 100) {
        
        std::vector<RooflinePoint> curve;
        curve.reserve(num_points);
        
        double peak_flops = getPeakFlops(precision);
        double peak_bandwidth = static_cast<double>(peak_.hbm_bandwidth_gbps);
        
        // Knee point (where memory bound meets compute bound)
        double knee_ai = peak_flops / peak_bandwidth;
        
        // Generate points
        for (uint32_t i = 0; i < num_points; ++i) {
            double log_ai = -2.0 + (4.0 * i / num_points);  // 0.01 to 100
            double ai = std::pow(10.0, log_ai);
            
            RooflinePoint pt;
            pt.arithmetic_intensity = ai;
            pt.performance = calculateRoofline(ai, precision);
            pt.region = ai < knee_ai ? "memory" : "compute";
            
            curve.push_back(pt);
        }
        
        return curve;
    }
    
    /**
     * @brief Optimize kernel for roofline
     */
    struct OptimizationSuggestion {
        std::string kernel_name;
        std::string issue;
        std::string recommendation;
        double potential_speedup;
    };
    
    std::vector<OptimizationSuggestion> suggestOptimizations(
        const std::vector<KernelCharacteristics>& kernels) {
        
        std::vector<OptimizationSuggestion> suggestions;
        
        double peak_flops = getPeakFlops("fp16");
        double peak_bandwidth = static_cast<double>(peak_.hbm_bandwidth_gbps);
        double knee_ai = peak_flops / peak_bandwidth;
        
        for (const auto& kernel : kernels) {
            OptimizationSuggestion sug;
            sug.kernel_name = kernel.name;
            
            if (kernel.bottleneck == "memory") {
                sug.issue = "Memory bound - low arithmetic intensity";
                
                if (kernel.arithmetic_intensity < knee_ai / 4) {
                    sug.recommendation = "Increase data reuse, use blocking/tiling";
                    sug.potential_speedup = 2.0;
                } else if (kernel.arithmetic_intensity < knee_ai / 2) {
                    sug.recommendation = "Use cache blocking, reduce memory accesses";
                    sug.potential_speedup = 1.5;
                } else {
                    sug.recommendation = "Optimize memory access pattern";
                    sug.potential_speedup = 1.2;
                }
            } else {
                sug.issue = "Compute bound";
                sug.recommendation = "Use lower precision (INT8) or increase parallelization";
                sug.potential_speedup = 1.3;
            }
            
            suggestions.push_back(sug);
        }
        
        return suggestions;
    }

private:
    double getPeakFlops(const std::string& precision) const {
        if (precision == "fp32") return peak_.fp32_tflops * 1e3;
        if (precision == "fp16") return peak_.fp16_tflops * 1e3;
        if (precision == "bf16") return peak_.bf16_tflops * 1e3;
        if (precision == "int8") return peak_.int8_tops * 1e3;
        if (precision == "int4") return peak_.int4_tops * 1e3;
        return peak_.fp16_tflops * 1e3;
    }
    
    PeakSpecifications peak_;
};

/**
 * @brief Performance bound analysis
 */
class BoundAnalyzer {
public:
    struct BoundResult {
        double compute_bound_percent;
        double memory_bound_percent;
        double cache_bound_percent;
        double communication_bound_percent;
        std::string primary_bottleneck;
        double theoretical_peak_tflops;
        double achievable_tflops;
        double efficiency_percent;
    };
    
    explicit BoundAnalyzer(const PeakSpecifications& peak) : peak_(peak) {}
    
    /**
     * @brief Analyze performance bounds for a kernel
     */
    BoundResult analyzeBounds(
        uint64_t flops,
        uint64_t hbm_bytes,
        uint64_t l2_bytes,
        uint64_t compute_cycles,
        uint64_t memory_cycles) {
        
        BoundResult result;
        
        double peak_flops = peak_.fp16_tflops * 1e3;
        double peak_hbm_bw = static_cast<double>(peak_.hbm_bandwidth_gbps);
        double peak_l2_bw = static_cast<double>(peak_.l2_bandwidth_gbps);
        
        // Time estimates
        double compute_time = compute_cycles / (peak_flops / 1e12);  // FLOPs -> seconds
        double hbm_time = hbm_bytes / (peak_hbm_bw * 1e9 / 8);     // bytes -> seconds
        double l2_time = l2_bytes / (peak_l2_bw * 1e9 / 8);
        
        double total_time = std::max({compute_time, hbm_time, l2_time});
        
        if (total_time > 0) {
            result.compute_bound_percent = (compute_time / total_time) * 100;
            result.memory_bound_percent = (hbm_time / total_time) * 100;
            result.cache_bound_percent = (l2_time / total_time) * 100;
        }
        
        // Determine primary bottleneck
        if (compute_time >= hbm_time && compute_time >= l2_time) {
            result.primary_bottleneck = "compute";
            result.theoretical_peak_tflops = peak_flops;
        } else if (hbm_time >= l2_time) {
            result.primary_bottleneck = "memory";
            result.theoretical_peak_tflops = peak_hbm_bw * 8 * peak_.hbm_bandwidth_gbps;
        } else {
            result.primary_bottleneck = "cache";
            result.theoretical_peak_tflops = peak_l2_bw * 8;
        }
        
        result.achievable_tflops = flops / (total_time * 1e12);
        result.efficiency_percent = (result.achievable_tflops / peak_flops) * 100;
        
        return result;
    }
    
    /**
     * @brief Estimate AI improvement needed
     */
    double estimateAIImprovement(
        const BoundResult& bound,
        double target_efficiency = 80.0) {
        
        double current_peak = bound.theoretical_peak_tflops;
        double current_eff = bound.efficiency_percent;
        
        if (current_eff >= target_efficiency) return 1.0;
        
        // How much more AI is needed?
        double peak_bandwidth = static_cast<double>(peak_.hbm_bandwidth_gbps);
        double peak_flops = peak_.fp16_tflops * 1e3;
        
        // At target efficiency, we'd achieve target_efficiency% of peak
        double target_ai = (peak_flops * target_efficiency / 100) / peak_bandwidth;
        
        // Current AI would need to increase by this factor
        // (simplified - actual calculation depends on current AI)
        return (target_efficiency / current_eff);
    }

private:
    PeakSpecifications peak_;
};

/**
 * @brief Multi-level roofline analyzer (HBM, L2, Registers)
 */
class MultiLevelRoofline {
public:
    MultiLevelRoofline(const PeakSpecifications& peak) : peak_(peak) {}
    
    struct RooflineLevel {
        std::string name;
        double bandwidth_tbps;     // TB/s
        double latency_ns;
        double capacity_mb;
    };
    
    struct AnalysisResult {
        std::vector<RooflinePoint> hbm_roofline;
        std::vector<RooflinePoint> l2_roofline;
        std::vector<RooflinePoint> reg_roofline;
        std::vector<KernelCharacteristics> kernel_analyses;
        std::vector<std::string> optimization_hints;
    };
    
    /**
     * @brief Generate multi-level roofline
     */
    AnalysisResult analyzeMultiLevel(
        const std::vector<std::tuple<std::string, uint64_t, uint64_t>>& kernels) {
        
        AnalysisResult result;
        
        // HBM roofline
        RooflineModel hbm_roofline(peak_);
        result.hbm_roofline = hbm_roofline.generateRooflineCurve("fp16", 100);
        
        // L2 roofline (higher bandwidth)
        PeakSpecifications l2_peak = peak_;
        l2_peak.hbm_bandwidth_gbps = peak_.l2_bandwidth_gbps;
        RooflineModel l2_roofline(l2_peak);
        result.l2_roofline = l2_roofline.generateRooflineCurve("fp16", 100);
        
        // Analyze each kernel
        for (const auto& [name, flops, bytes] : kernels) {
            auto analysis = hbm_roofline.analyzeKernel(name, flops, bytes);
            result.kernel_analyses.push_back(analysis);
        }
        
        // Generate hints
        RooflineModel rm(peak_);
        auto suggestions = rm.suggestOptimizations(result.kernel_analyses);
        for (const auto& sug : suggestions) {
            result.optimization_hints.push_back(
                sug.kernel_name + ": " + sug.recommendation);
        }
        
        return result;
    }
    
    /**
     * @brief Generate ASCII visualization
     */
    std::string visualize(
        const std::vector<KernelCharacteristics>& kernels,
        uint32_t width = 80,
        uint32_t height = 30) {
        
        std::ostringstream oss;
        
        double max_ai = 100.0;
        double max_tflops = peak_.fp16_tflops;
        
        // Create grid
        std::vector<std::string> grid(height, std::string(width, ' '));
        
        // Plot roofline
        for (uint32_t x = 0; x < width; ++x) {
            double ai = max_ai * x / width;
            double roofline = peak_.fp16_tflops;
            double memory_bound = ai * peak_.hbm_bandwidth_gbps / 8;
            double y = height - 1 - (std::min(roofline, memory_bound) / max_tflops * height);
            
            if (y >= 0 && y < height) {
                grid[static_cast<size_t>(y)][x] = '*';
            }
        }
        
        // Plot kernels
        for (const auto& kernel : kernels) {
            if (kernel.arithmetic_intensity <= max_ai && 
                kernel.achieved_tflops <= max_tflops) {
                size_t x = static_cast<size_t>(
                    kernel.arithmetic_intensity / max_ai * width);
                size_t y = static_cast<size_t>(
                    height - 1 - (kernel.achieved_tflops / max_tflops * height));
                
                if (x < width && y < height) {
                    grid[y][x] = 'o';
                }
            }
        }
        
        // Output
        for (const auto& row : grid) {
            oss << row << "\n";
        }
        
        return oss.str();
    }

private:
    PeakSpecifications peak_;
    
    // RooflinePoint struct (same as in RooflineModel)
    struct RooflinePoint {
        double arithmetic_intensity;
        double performance;
        std::string region;
    };
};

} // namespace analysis
} // namespace cortex
