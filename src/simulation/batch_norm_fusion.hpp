/**
 * @file batch_norm_fusion.hpp
 * @brief Batch Normalization Fusion for Hardware Accelerators
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <algorithm>

namespace cortex {
namespace simulation {

/**
 * @brief Batch normalization parameters
 */
struct BatchNormParams {
    std::vector<float> gamma;    // Scale
    std::vector<float> beta;     // Offset
    std::vector<float> mean;
    std::vector<float> variance;
    float epsilon;
    uint32_t num_channels;
    bool fused;                  // Already fused with preceding layer
    
    BatchNormParams() : epsilon(1e-5f), num_channels(0), fused(false) {}
};

/**
 * @brief Layer normalization parameters
 */
struct LayerNormParams {
    std::vector<float> gamma;
    std::vector<float> beta;
    float epsilon;
    uint32_t hidden_size;
    
    LayerNormParams() : epsilon(1e-5f), hidden_size(0) {}
};

/**
 * @brief Fusion pattern types
 */
enum class FusionPattern {
    CONV_BN,
    LINEAR_BN,
    BN_RELU,
    CONV_BN_RELU,
    MULTI_HEAD_ATTENTION,
    FEEDFORWARD_NETWORK,
    RESIDUAL_BLOCK,
    SKIP_CONNECTION,
    CUSTOM
};

/**
 * @brief Fused operation descriptor
 */
struct FusedOperation {
    std::string name;
    FusionPattern pattern;
    std::vector<std::string> original_ops;
    std::vector<float> fused_weights;
    std::vector<float> fused_bias;
    
    // Efficiency metrics
    uint64_t memory_saved_bytes;
    uint64_t cycles_saved;
    double speedup_factor;
    
    FusedOperation() : memory_saved_bytes(0), cycles_saved(0), speedup_factor(1.0) {}
};

/**
 * @brief Batch normalization fusion analyzer
 */
class BatchNormFusion {
public:
    BatchNormFusion() : fusion_enabled_(true), aggressive_fusion_(false) {}
    
    /**
     * @brief Fuse batch norm with preceding convolution
     */
    struct FusionResult {
        std::vector<float> fused_weights;
        std::vector<float> fused_bias;
        uint64_t memory_saved_bytes;
        uint64_t cycles_saved;
        double latency_improvement_percent;
    };
    
    /**
     * @brief Fuse conv layer with batch normalization
     */
    FusionResult fuseConvBN(const std::vector<float>& conv_weight,
                           const std::vector<float>& conv_bias,
                           const BatchNormParams& bn,
                           const std::array<int, 4>& conv_shape) {
        FusionResult result;
        
        // BN formula: y = gamma * (x - mean) / sqrt(variance + eps) + beta
        // When fused with conv: y = gamma/sqrt(var+eps) * (conv(x) - mean) + beta
        //                      = gamma/sqrt(var+eps) * conv(x) + (beta - gamma*mean/sqrt(var+eps))
        
        uint32_t out_channels = conv_shape[0];
        uint32_t in_channels = conv_shape[1];
        uint32_t kernel_h = conv_shape[2];
        uint32_t kernel_w = conv_shape[3];
        
        result.fused_weights.resize(conv_weight.size());
        
        // Scale conv weights by BN scale factor
        for (uint32_t oc = 0; oc < out_channels; ++oc) {
            float bn_scale = bn.gamma[oc] / std::sqrt(bn.variance[oc] + bn.epsilon);
            
            uint32_t weight_offset = oc * in_channels * kernel_h * kernel_w;
            for (uint32_t i = 0; i < in_channels * kernel_h * kernel_w; ++i) {
                result.fused_weights[weight_offset + i] = conv_weight[weight_offset + i] * bn_scale;
            }
        }
        
        // Fuse bias
        result.fused_bias.resize(out_channels);
        for (uint32_t oc = 0; oc < out_channels; ++oc) {
            float bn_scale = bn.gamma[oc] / std::sqrt(bn.variance[oc] + bn.epsilon);
            float bn_offset = bn.beta[oc] - bn_scale * bn.mean[oc];
            result.fused_bias[oc] = (conv_bias.empty() ? 0.0f : conv_bias[oc]) * bn_scale + bn_offset;
        }
        
        // Calculate savings
        uint64_t original_memory = (conv_weight.size() + conv_bias.size()) * sizeof(float);
        uint64_t fused_memory = result.fused_weights.size() * sizeof(float) + 
                                result.fused_bias.size() * sizeof(float);
        result.memory_saved_bytes = original_memory - fused_memory;
        
        // Estimate cycles saved (BN is absorbed into conv)
        uint64_t bn_cycles = out_channels * 3;  // mean, var, normalize
        uint64_t activation_cycles = out_channels * 2;  // scale and add bias
        result.cycles_saved = bn_cycles + activation_cycles;
        
        result.latency_improvement_percent = 
            (double)result.cycles_saved / (result.cycles_saved + out_channels * in_channels * kernel_h * kernel_w) * 100;
        
        return result;
    }
    
    /**
     * @brief Fuse linear layer with batch normalization
     */
    FusionResult fuseLinearBN(const std::vector<float>& linear_weight,
                              const std::vector<float>& linear_bias,
                              const BatchNormParams& bn) {
        FusionResult result;
        
        uint32_t out_features = bn.num_channels;
        uint32_t in_features = linear_weight.size() / out_features;
        
        result.fused_weights.resize(linear_weight.size());
        
        // Scale weights
        for (uint32_t oc = 0; oc < out_features; ++oc) {
            float bn_scale = bn.gamma[oc] / std::sqrt(bn.variance[oc] + bn.epsilon);
            uint32_t row_offset = oc * in_features;
            for (uint32_t ic = 0; ic < in_features; ++ic) {
                result.fused_weights[row_offset + ic] = linear_weight[row_offset + ic] * bn_scale;
            }
        }
        
        // Fuse bias
        result.fused_bias.resize(out_features);
        for (uint32_t oc = 0; oc < out_features; ++oc) {
            float bn_scale = bn.gamma[oc] / std::sqrt(bn.variance[oc] + bn.epsilon);
            float bn_offset = bn.beta[oc] - bn_scale * bn.mean[oc];
            result.fused_bias[oc] = (linear_bias.empty() ? 0.0f : linear_bias[oc]) * bn_scale + bn_offset;
        }
        
        result.memory_saved_bytes = sizeof(float) * (bn.gamma.size() + bn.beta.size() + 
                                                       bn.mean.size() + bn.variance.size());
        result.cycles_saved = out_features * 6;  // Simplified
        result.latency_improvement_percent = 15.0;  // Typical
        
        return result;
    }
    
    /**
     * @brief Analyze fusion opportunities in a model
     */
    struct FusionAnalysis {
        uint32_t num_conv_fusions;
        uint32_t num_linear_fusions;
        uint64_t total_memory_saved;
        uint64_t total_cycles_saved;
        double estimated_speedup;
        std::vector<FusedOperation> fusion_candidates;
    };
    
    FusionAnalysis analyzeFusionOpportunities(
        const std::vector<std::array<int, 4>>& conv_shapes,
        const std::vector<std::vector<float>>& conv_weights,
        const std::vector<BatchNormParams>& bn_params) {
        
        FusionAnalysis analysis;
        
        uint32_t num_fusions = std::min({conv_shapes.size(), 
                                          conv_weights.size(), 
                                          bn_params.size()});
        
        for (uint32_t i = 0; i < num_fusions; ++i) {
            FusedOperation fused_op;
            fused_op.name = "conv" + std::to_string(i) + "_bn" + std::to_string(i);
            fused_op.pattern = FusionPattern::CONV_BN;
            fused_op.original_ops = {"conv" + std::to_string(i), "bn" + std::to_string(i)};
            
            auto result = fuseConvBN(conv_weights[i], {}, bn_params[i], conv_shapes[i]);
            
            fused_op.fused_weights = result.fused_weights;
            fused_op.fused_bias = result.fused_bias;
            fused_op.memory_saved_bytes = result.memory_saved_bytes;
            fused_op.cycles_saved = result.cycles_saved;
            fused_op.speedup_factor = 1.0 + result.latency_improvement_percent / 100.0;
            
            analysis.fusion_candidates.push_back(fused_op);
            analysis.num_conv_fusions++;
            analysis.total_memory_saved += result.memory_saved_bytes;
            analysis.total_cycles_saved += result.cycles_saved;
        }
        
        // Calculate total speedup
        uint64_t total_cycles = 0;
        for (const auto& op : analysis.fusion_candidates) {
            total_cycles += op.cycles_saved;
        }
        analysis.estimated_speedup = 1.0 + 
            (double)analysis.total_cycles_saved / 
            std::max(1ULL, total_cycles);
        
        return analysis;
    }
    
    void setFusionEnabled(bool enabled) { fusion_enabled_ = enabled; }
    void setAggressiveFusion(bool aggressive) { aggressive_fusion_ = aggressive; }
    bool isFusionEnabled() const { return fusion_enabled_; }

private:
    bool fusion_enabled_;
    bool aggressive_fusion_;
};

/**
 * @brief Layer normalization fusion
 */
class LayerNormFusion {
public:
    /**
     * @brief Fuse layer norm into preceding operation
     */
    struct LayerNormFusionResult {
        std::vector<float> additive_bias;
        std::vector<float> multiplicative_weight;
        uint64_t cycles_saved;
        bool fusion_possible;
    };
    
    LayerNormFusionResult analyzeLayerNormFusion(
        const LayerNormParams& ln,
        const std::string& preceding_op) {
        
        LayerNormFusionResult result;
        result.fusion_possible = false;
        
        // Layer norm can be fused with residual addition
        if (preceding_op == "residual_add") {
            result.fusion_possible = true;
            
            // Compute effective bias for fused residual + layer norm
            result.additive_bias.resize(ln.hidden_size);
            result.multiplicative_weight = ln.gamma;
            
            // Simplified: just store the ln weights
            result.cycles_saved = ln.hidden_size * 2;
        }
        
        return result;
    }
};

/**
 * @brief Residual block fusion analyzer
 */
class ResidualFusion {
public:
    struct ResidualFusionResult {
        uint32_t num_fusions;
        uint64_t memory_saved;
        uint64_t bandwidth_reduction_bytes;
        double compute_reduction_percent;
        std::vector<std::string> fused_blocks;
    };
    
    ResidualFusionResult analyzeResidualFusion(
        uint32_t num_residual_blocks,
        const std::vector<uint64_t>& block_compute_cycles,
        const std::vector<uint64_t>& block_memory_bytes) {
        
        ResidualFusionResult result;
        result.num_fusions = num_residual_blocks;
        
        // Each residual block can eliminate one memory read/write pair
        for (size_t i = 0; i < num_residual_blocks; ++i) {
            result.memory_saved += block_memory_bytes[i] / 2;  // Partial savings
            result.bandwidth_reduction_bytes += block_memory_bytes[i];
        }
        
        // Compute reduction from fusion
        uint64_t total_compute = 0;
        for (uint64_t cycles : block_compute_cycles) {
            total_compute += cycles;
        }
        
        // Fusion typically saves 10-20% of residual block compute
        result.compute_reduction_percent = 15.0;
        
        return result;
    }
};

/**
 * @brief Generic fusion optimizer
 */
class FusionOptimizer {
public:
    FusionOptimizer() : enable_conv_bn_(true), enable_residual_(true), 
                        enable_layer_norm_(true) {}
    
    /**
     * @brief Optimize model by applying all fusion patterns
     */
    struct OptimizationResult {
        uint32_t total_fusions_applied;
        uint64_t memory_reduction_bytes;
        uint64_t compute_reduction_cycles;
        double latency_reduction_percent;
        double throughput_improvement_percent;
        std::vector<std::string> fusion_summary;
    };
    
    OptimizationResult optimize(
        const std::vector<FusionPattern>& patterns) {
        
        OptimizationResult result;
        
        for (auto pattern : patterns) {
            switch (pattern) {
                case FusionPattern::CONV_BN:
                case FusionPattern::LINEAR_BN:
                case FusionPattern::BN_RELU:
                case FusionPattern::CONV_BN_RELU:
                    result.total_fusions_applied++;
                    result.memory_reduction_bytes += 1024;  // Estimated
                    result.compute_reduction_cycles += 1000;
                    result.fusion_summary.push_back("BatchNorm fusion");
                    break;
                    
                case FusionPattern::RESIDUAL_BLOCK:
                    result.total_fusions_applied++;
                    result.memory_reduction_bytes += 2048;
                    result.compute_reduction_cycles += 2000;
                    result.fusion_summary.push_back("Residual fusion");
                    break;
                    
                default:
                    break;
            }
        }
        
        // Calculate improvements
        uint64_t total_ops = result.compute_reduction_cycles * 10;  // Estimate
        result.latency_reduction_percent = 
            (double)result.compute_reduction_cycles / total_ops * 100;
        result.throughput_improvement_percent = 
            result.latency_reduction_percent / (100 - result.latency_reduction_percent) * 100;
        
        return result;
    }
    
    void enableConvBN(bool enable) { enable_conv_bn_ = enable; }
    void enableResidual(bool enable) { enable_residual_ = enable; }
    void enableLayerNorm(bool enable) { enable_layer_norm_ = enable; }

private:
    bool enable_conv_bn_;
    bool enable_residual_;
    bool enable_layer_norm_;
};

} // namespace simulation
} // namespace cortex
