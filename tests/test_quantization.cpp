/**
 * @file test_quantization.cpp
 * @brief Tests for Quantization Module
 */

#include <gtest/gtest.h>
#include "simulation/quantization.hpp"

using namespace cortex::simulation;

class QuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        calibrator_ = std::make_unique<QuantizationCalibrator>();
        
        // Generate test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        test_data_.resize(1000);
        for (auto& val : test_data_) {
            val = dist(gen);
        }
    }
    
    std::unique_ptr<QuantizationCalibrator> calibrator_;
    std::vector<float> test_data_;
};

TEST_F(QuantizationTest, Calibration) {
    calibrator_->collectStats(test_data_);
    
    auto params = calibrator_->computeParams(Precision::INT8);
    
    EXPECT_GT(params.calibration_samples, 0);
    EXPECT_EQ(params.params.precision, Precision::INT8);
    EXPECT_GT(params.observed_min, 0);
    EXPECT_LT(params.observed_max, 0);
}

TEST_F(QuantizationTest, SymmetricQuantization) {
    calibrator_->setScheme(QuantScheme::SYMMETRIC);
    calibrator_->collectStats(test_data_);
    
    auto params = calibrator_->computeParams(Precision::INT8);
    EXPECT_EQ(params.params.scheme, QuantScheme::SYMMETRIC);
}

TEST_F(QuantizationTest, AsymmetricQuantization) {
    calibrator_->setScheme(QuantScheme::ASYMMETRIC);
    calibrator_->collectStats(test_data_);
    
    auto params = calibrator_->computeParams(Precision::INT8);
    EXPECT_EQ(params.params.scheme, QuantScheme::ASYMMETRIC);
}

TEST_F(QuantizationTest, Quantizer) {
    QuantParams params;
    params.scheme = QuantScheme::SYMMETRIC;
    params.precision = Precision::INT8;
    params.scale = 0.01f;
    
    Quantizer quantizer(params);
    
    int8_t q = quantizer.quantize(1.0f);
    EXPECT_GE(q, -127);
    EXPECT_LE(q, 127);
    
    float dq = quantizer.dequantize(q);
    EXPECT_NEAR(dq, 1.0f, 0.1f);
}

TEST_F(QuantizationTest, QuantizeVector) {
    QuantParams params;
    params.scheme = QuantScheme::SYMMETRIC;
    params.precision = Precision::INT8;
    params.scale = 0.1f;
    
    Quantizer quantizer(params);
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto quantized = quantizer.quantize(data);
    
    EXPECT_EQ(quantized.size(), data.size());
}

TEST_F(QuantizationTest, Float16Conversion) {
    float original = 3.14159f;
    
    uint16_t fp16 = Float16Converter::fp32ToFp16(original);
    float recovered = Float16Converter::fp16ToFp32(fp16);
    
    EXPECT_NEAR(recovered, original, 0.001f);
}

TEST_F(QuantizationTest, BF16Conversion) {
    float original = 2.71828f;
    
    uint16_t bf16 = Float16Converter::fp32ToBf16(original);
    float recovered = Float16Converter::bf16ToFp32(bf16);
    
    EXPECT_NEAR(recovered, original, 0.01f);
}

TEST_F(QuantizationTest, ErrorAnalysis) {
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> quantized = {1.1f, 2.1f, 2.9f, 4.1f, 4.9f};
    
    auto error = QuantizationErrorAnalyzer::analyze(original, quantized);
    
    EXPECT_GT(error.snr_db, 0);
    EXPECT_GE(error.mse, 0);
}

TEST_F(QuantizationTest, ResetCalibrator) {
    calibrator_->collectStats(test_data_);
    calibrator_->reset();
    
    // After reset, should be able to start fresh
    calibrator_->collectStats({1.0f, 2.0f});
    auto params = calibrator_->computeParams(Precision::INT8);
    EXPECT_EQ(params.calibration_samples, 2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
