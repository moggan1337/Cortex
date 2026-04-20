/**
 * @file test_roofline.cpp
 * @brief Tests for Roofline Model Analysis
 */

#include <gtest/gtest.h>
#include "analysis/roofline_model.hpp"

using namespace cortex::analysis;

class RooflineTest : public ::testing::Test {
protected:
    void SetUp() override {
        peak_specs_.fp16_tflops = 180;
        peak_specs_.hbm_bandwidth_gbps = 900;
        roofline_ = std::make_unique<RooflineModel>(peak_specs_);
    }
    
    PeakSpecifications peak_specs_;
    std::unique_ptr<RooflineModel> roofline_;
};

TEST_F(RooflineTest, RooflineCalculation) {
    // High AI should hit compute bound
    double high_ai = 100.0;
    double perf = roofline_->calculateRoofline(high_ai, "fp16");
    
    double peak_tflops = peak_specs_.fp16_tflops * 1000;  // Convert to GFLOPS
    EXPECT_NEAR(perf, peak_tflops, 1.0);
}

TEST_F(RooflineTest, MemoryBoundRegion) {
    // Low AI should be memory bound
    double low_ai = 0.1;
    double perf = roofline_->calculateRoofline(low_ai, "fp16");
    
    double expected = low_ai * peak_specs_.hbm_bandwidth_gbps;
    EXPECT_NEAR(perf, expected, 1.0);
}

TEST_F(RooflineTest, KernelAnalysis) {
    auto kernel = roofline_->analyzeKernel("test_kernel", 
                                            1'000'000'000ULL,  // FLOPs
                                            10'000'000ULL,     // Bytes
                                            "fp16");
    
    EXPECT_EQ(kernel.name, "test_kernel");
    EXPECT_GT(kernel.arithmetic_intensity, 0);
    EXPECT_GT(kernel.achieved_tflops, 0);
}

TEST_F(RooflineTest, EfficiencyCalculation) {
    double achieved = 90.0;  // GFLOPS
    double efficiency = roofline_->calculateEfficiency(achieved, "fp16");
    
    double expected = (achieved / (peak_specs_.fp16_tflops * 1000)) * 100;
    EXPECT_NEAR(efficiency, expected, 0.1);
}

TEST_F(RooflineTest, RooflineCurve) {
    auto curve = roofline_->generateRooflineCurve("fp16", 50);
    
    EXPECT_EQ(curve.size(), 50);
    
    // First point should be memory bound, last should be compute bound
    EXPECT_EQ(curve.front().region, "memory");
    EXPECT_EQ(curve.back().region, "compute");
}

TEST_F(RooflineTest, OptimizationSuggestions) {
    std::vector<KernelCharacteristics> kernels = {
        {"conv1", 5.0, 50.0, 40.0, 50.0, "memory"},
        {"conv2", 50.0, 100.0, 90.0, 100.0, "compute"}
    };
    
    auto suggestions = roofline_->suggestOptimizations(kernels);
    
    EXPECT_EQ(suggestions.size(), 2);
    EXPECT_TRUE(suggestions[0].recommendation.find("blocking") != std::string::npos ||
                suggestions[0].recommendation.find("reuse") != std::string::npos);
}

TEST_F(RooflineTest, BoundAnalyzer) {
    BoundAnalyzer analyzer(peak_specs_);
    
    auto bounds = analyzer.analyzeBounds(
        1'000'000'000ULL,  // FLOPs
        100'000'000ULL,    // HBM bytes
        50'000'000ULL,     // L2 bytes
        5'000'000ULL,      // Compute cycles
        10'000'000ULL      // Memory cycles
    );
    
    EXPECT_GE(bounds.compute_bound_percent + bounds.memory_bound_percent, 90.0);
    EXPECT_FALSE(bounds.primary_bottleneck.empty());
}

TEST_F(RooflineTest, PeakSpecifications) {
    PeakSpecifications specs;
    specs.fp32_tflops = 90;
    specs.fp16_tflops = 180;
    specs.bf16_tflops = 180;
    specs.int8_tops = 360;
    specs.hbm_bandwidth_gbps = 900;
    
    RooflineModel rm(specs);
    
    double fp32_roof = rm.calculateRoofline(10.0, "fp32");
    double fp16_roof = rm.calculateRoofline(10.0, "fp16");
    
    EXPECT_GT(fp16_roof, fp32_roof);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
