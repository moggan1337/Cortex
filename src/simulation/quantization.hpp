/**
 * @file quantization.hpp
 * @brief Quantization Support for INT8, FP16, BF16
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
#include <numeric>
#include <random>
#include <cstdint>

namespace cortex {
namespace simulation {

/**
 * @brief Quantization schemes
 */
enum class QuantScheme {
    NONE,
    SYMMETRIC,
    ASYMMETRIC,
    AFFINE,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP,
    DYNAMIC,
    STATIC,
    MIXED_PRECISION
};

/**
 * @brief Data precision types
 */
enum class Precision {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,   // 4-bit exponent, 3-bit mantissa
    FP8_E5M2,   // 5-bit exponent, 2-bit mantissa
    INT8,
    INT4,
    INT2,
    UINT8,
    UINT4
};

/**
 * @brief Quantization parameters
 */
struct QuantParams {
    QuantScheme scheme;
    Precision precision;
    
    // Scale factors
    float scale;
    std::vector<float> scales;     // Per-channel scales
    std::vector<float> zero_points; // For asymmetric quantization
    
    // Range
    float min_val;
    float max_val;
    float range;
    
    // Group quantization
    uint32_t group_size;
    
    // Calibration
    std::string calibration_method;
    float percentile;
    
    QuantParams() : scheme(QuantScheme::SYMMETRIC), precision(Precision::INT8),
                    scale(1.0f), min_val(0), max_val(0), range(0), group_size(128),
                    calibration_method("max"), percentile(99.99f) {}
};

/**
 * @brief Calibrated quantization parameters
 */
struct CalibratedQuantParams {
    QuantParams params;
    float observed_min;
    float observed_max;
    float mean;
    float std_dev;
    uint64_t calibration_samples;
    
    CalibratedQuantParams() : observed_min(0), observed_max(0), 
                              mean(0), std_dev(0), calibration_samples(0) {}
};

/**
 * @brief Quantization calibrator
 */
class QuantizationCalibrator {
public:
    explicit QuantizationCalibrator(QuantScheme scheme = QuantScheme::SYMMETRIC)
        : scheme_(scheme), percentile_(99.99f) {}
    
    /**
     * @brief Collect statistics during calibration
     */
    void collectStats(const std::vector<float>& data) {
        if (data.empty()) return;
        
        // Update running statistics
        for (float val : data) {
            hist_[quantizeHist(val)]++;
            sum_ += val;
            sum_sq_ += val * val;
            count_++;
            min_ = std::min(min_, val);
            max_ = std::max(max_, val);
        }
    }
    
    /**
     * @brief Compute calibration parameters
     */
    CalibratedQuantParams computeParams(Precision target_precision) {
        CalibratedQuantParams result;
        result.params.scheme = scheme_;
        result.params.precision = target_precision;
        result.observed_min = min_;
        result.observed_max = max_;
        result.mean = sum_ / std::max(1ULL, count_);
        result.std_dev = std::sqrt(sum_sq_ / std::max(1ULL, count_) - result.mean * result.mean);
        result.calibration_samples = count_;
        
        int num_levels = getNumLevels(target_precision);
        
        switch (scheme_) {
            case QuantScheme::SYMMETRIC: {
                float abs_max = std::max(std::abs(min_), std::abs(max_));
                result.params.scale = abs_max / (num_levels / 2 - 1);
                result.params.zero_points.push_back(0);
                break;
            }
            
            case QuantScheme::ASYMMETRIC: {
                result.params.scale = (max_ - min_) / num_levels;
                result.params.zero_point = -min_ / result.params.scale;
                break;
            }
            
            case QuantScheme::PER_CHANNEL: {
                // Simplified: would need channel-aware data
                result.params.scale = (max_ - min_) / num_levels;
                result.params.scales.resize(1, result.params.scale);
                break;
            }
            
            case QuantScheme::MIXED_PRECISION: {
                // Use different precisions for different layers
                result.params.scale = (max_ - min_) / num_levels;
                break;
            }
            
            default:
                result.params.scale = 1.0f;
        }
        
        result.params.min_val = min_;
        result.params.max_val = max_;
        result.params.range = max_ - min_;
        result.params.calibration_method = "entropy";  // Default
        result.params.percentile = percentile_;
        
        return result;
    }
    
    /**
     * @brief Reset calibrator state
     */
    void reset() {
        hist_.fill(0);
        sum_ = 0;
        sum_sq_ = 0;
        count_ = 0;
        min_ = std::numeric_limits<float>::max();
        max_ = std::numeric_limits<float>::lowest();
    }
    
    void setPercentile(float p) { percentile_ = p; }
    void setScheme(QuantScheme s) { scheme_ = s; }

private:
    int quantizeHist(float val) const {
        // Simplified histogram binning
        int max_bin = 2047;
        float normalized = (val - min_) / (max_ - min_ + 1e-8);
        return std::clamp(static_cast<int>(normalized * max_bin), 0, max_bin);
    }
    
    int getNumLevels(Precision p) const {
        switch (p) {
            case Precision::FP16:
            case Precision::BF16:
                return 65536;  // 16-bit
            case Precision::INT8:
                return 256;
            case Precision::INT4:
                return 16;
            case Precision::INT2:
                return 4;
            case Precision::UINT8:
                return 256;
            case Precision::UINT4:
                return 16;
            default:
                return 256;
        }
    }
    
    QuantScheme scheme_;
    float percentile_;
    std::array<uint64_t, 2048> hist_;
    double sum_;
    double sum_sq_;
    uint64_t count_;
    float min_;
    float max_;
    
    // Asymmetric params
    float zero_point = 0;
};

/**
 * @brief Quantizer class
 */
class Quantizer {
public:
    Quantizer() : params_() {}
    
    explicit Quantizer(const QuantParams& params) : params_(params) {}
    
    /**
     * @brief Quantize a single value
     */
    int8_t quantize(float val) {
        if (params_.scheme == QuantScheme::SYMMETRIC) {
            float scaled = val / params_.scale;
            return std::clamp(static_cast<int8_t>(std::round(scaled)), 
                            -127, 127);
        } else {
            float scaled = (val - params_.min_val) / params_.scale;
            return std::clamp(static_cast<int8_t>(std::round(scaled)), 0, 255);
        }
    }
    
    /**
     * @brief Quantize a vector of values
     */
    std::vector<int8_t> quantize(const std::vector<float>& data) {
        std::vector<int8_t> result;
        result.reserve(data.size());
        
        for (float val : data) {
            result.push_back(quantize(val));
        }
        
        return result;
    }
    
    /**
     * @brief Dequantize a single value
     */
    float dequantize(int8_t val) const {
        if (params_.scheme == QuantScheme::SYMMETRIC) {
            return val * params_.scale;
        } else {
            return val * params_.scale + params_.min_val;
        }
    }
    
    /**
     * @brief Dequantize a vector
     */
    std::vector<float> dequantize(const std::vector<int8_t>& data) const {
        std::vector<float> result;
        result.reserve(data.size());
        
        for (int8_t val : data) {
            result.push_back(dequantize(val));
        }
        
        return result;
    }
    
    void setParams(const QuantParams& params) { params_ = params; }
    const QuantParams& getParams() const { return params_; }

private:
    QuantParams params_;
};

/**
 * @brief FP16/BF16 converter
 */
class Float16Converter {
public:
    /**
     * @brief Convert FP32 to FP16
     */
    static uint16_t fp32ToFp16(float f) {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(float));
        
        uint16_t sign = (bits >> 16) & 0x8000;
        int16_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint16_t mantissa = (bits >> 13) & 0x3FF;
        
        if (exp <= 0) {
            // Denormal or zero
            return sign;
        } else if (exp >= 31) {
            // Inf or NaN
            return sign | 0x7C00 | (exp == 31 && (bits & 0x7FFFFF) ? 0x200 : 0);
        }
        
        return sign | (exp << 10) | mantissa;
    }
    
    /**
     * @brief Convert FP16 to FP32
     */
    static float fp16ToFp32(uint16_t h) {
        uint32_t bits = 0;
        
        uint32_t sign = (h & 0x8000) << 16;
        uint16_t exp = (h >> 10) & 0x1F;
        uint16_t mantissa = h & 0x3FF;
        
        if (exp == 0) {
            // Denormal
            if (mantissa == 0) {
                std::memcpy(&bits, &sign, sizeof(float));
                float result;
                std::memcpy(&result, &bits, sizeof(float));
                return result;
            }
            // Shift mantissa
            exp = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF;
        } else if (exp == 31) {
            // Inf or NaN
            bits = sign | 0x7F800000 | (mantissa << 13);
            float result;
            std::memcpy(&result, &bits, sizeof(float));
            return result;
        }
        
        bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
        
        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }
    
    /**
     * @brief Convert FP32 to BF16 (Brain Float)
     */
    static uint16_t fp32ToBf16(float f) {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(float));
        
        // BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
        return (bits >> 16) & 0xFFFF;
    }
    
    /**
     * @brief Convert BF16 to FP32
     */
    static float bf16ToFp32(uint16_t b) {
        uint32_t bits = (b << 16);
        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }
};

/**
 * @brief Quantization error analyzer
 */
class QuantizationErrorAnalyzer {
public:
    struct ErrorMetrics {
        float mean_error;
        float max_abs_error;
        float mse;
        float rmse;
        float rel_error_percent;
        double snr_db;
        float skeleton_error;  // Critical layer error
        int8_t saturations_high;
        int8_t saturations_low;
        double saturation_rate;
    };
    
    /**
     * @brief Analyze quantization error
     */
    static ErrorMetrics analyze(const std::vector<float>& original,
                                const std::vector<float>& quantized) {
        ErrorMetrics metrics;
        
        if (original.size() != quantized.size()) return metrics;
        
        size_t n = original.size();
        double sum_error = 0;
        double sum_sq_error = 0;
        double sum_original_sq = 0;
        
        metrics.max_abs_error = 0;
        
        for (size_t i = 0; i < n; ++i) {
            float error = quantized[i] - original[i];
            float abs_error = std::abs(error);
            
            sum_error += error;
            sum_sq_error += error * error;
            sum_original_sq += original[i] * original[i];
            
            metrics.max_abs_error = std::max(metrics.max_abs_error, abs_error);
            
            // Track saturations
            float rel_error = original[i] != 0 ? 
                              abs_error / std::abs(original[i]) : 0;
            if (rel_error > 0.1f) {  // 10% error threshold
                metrics.saturations_high++;
            }
        }
        
        metrics.mse = sum_sq_error / n;
        metrics.rmse = std::sqrt(metrics.mse);
        metrics.mean_error = sum_error / n;
        metrics.rel_error_percent = std::sqrt(sum_original_sq / n) > 0 ?
            (metrics.rmse / std::sqrt(sum_original_sq / n)) * 100 : 0;
        
        // SNR
        if (sum_original_sq > 0) {
            metrics.snr_db = 10 * std::log10(sum_original_sq / sum_sq_error);
        }
        
        metrics.saturation_rate = (double)metrics.saturations_high / n * 100;
        
        return metrics;
    }
    
    /**
     * @brief Estimate accuracy degradation from quantization
     */
    static float estimateAccuracyDegradation(const ErrorMetrics& error) {
        // Simplified model: higher error correlates with accuracy loss
        float base_loss = 0.0f;
        
        if (error.snr_db < 10) {
            base_loss += 5.0f;  // Severe quantization
        } else if (error.snr_db < 20) {
            base_loss += 2.0f;  // Moderate
        } else if (error.snr_db < 30) {
            base_loss += 0.5f;  // Mild
        }
        
        if (error.saturation_rate > 5.0) {
            base_loss += 3.0f;  // High saturation
        }
        
        return base_loss;
    }
};

/**
 * @brief Mixed precision optimizer
 */
class MixedPrecisionOptimizer {
public:
    struct LayerPrecision {
        std::string layer_name;
        Precision precision;
        QuantParams quant_params;
        float sensitivity_score;  // How sensitive to quantization
    };
    
    struct MixedPrecisionConfig {
        std::vector<LayerPrecision> layer_configs;
        uint32_t total_bits_saved;
        double estimated_speedup;
        float estimated_accuracy_loss;
    };
    
    /**
     * @brief Analyze layer sensitivity to quantization
     */
    static float analyzeLayerSensitivity(
        const std::vector<float>& activations,
        const std::vector<float>& gradients) {
        
        // Simple sensitivity based on gradient magnitude
        float grad_sum = 0;
        for (float g : gradients) {
            grad_sum += std::abs(g);
        }
        
        return grad_sum / std::max(1.0f, static_cast<float>(gradients.size()));
    }
    
    /**
     * @brief Optimize precision assignment
     */
    MixedPrecisionConfig optimize(
        const std::vector<std::string>& layer_names,
        const std::vector<std::vector<float>>& activations,
        const std::vector<std::vector<float>>& gradients,
        Precision high_precision = Precision::FP16,
        Precision low_precision = Precision::INT8) {
        
        MixedPrecisionConfig config;
        
        for (size_t i = 0; i < layer_names.size(); ++i) {
            LayerPrecision lp;
            lp.layer_name = layer_names[i];
            lp.sensitivity_score = analyzeLayerSensitivity(
                activations[i], gradients[i]);
            
            // Assign precision based on sensitivity
            if (lp.sensitivity_score > 1.0f) {
                lp.precision = high_precision;
            } else {
                lp.precision = low_precision;
            }
            
            config.layer_configs.push_back(lp);
        }
        
        // Calculate savings
        uint32_t high_bits = getBits(high_precision);
        uint32_t low_bits = getBits(low_precision);
        uint32_t diff_bits = high_bits - low_bits;
        
        config.total_bits_saved = 0;
        for (const auto& lp : config.layer_configs) {
            if (lp.precision == low_precision) {
                config.total_bits_saved += diff_bits;
            }
        }
        
        config.estimated_speedup = 1.0 + 
            (config.total_bits_saved / 1000.0) * 0.1;  // Simplified
        config.estimated_accuracy_loss = 1.5f;  // Estimated
        
        return config;
    }

private:
    static uint32_t getBits(Precision p) {
        switch (p) {
            case Precision::FP32: return 32;
            case Precision::FP16:
            case Precision::BF16: return 16;
            case Precision::INT8:
            case Precision::UINT8: return 8;
            case Precision::INT4:
            case Precision::UINT4: return 4;
            default: return 32;
        }
    }
};

} // namespace simulation
} // namespace cortex
