/**
 * @file test_systolic_array.cpp
 * @brief Tests for Systolic Array Module
 */

#include <gtest/gtest.h>
#include "core/systolic_array.hpp"

using namespace cortex::core;

class SystolicArrayTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = SystolicArrayConfig(8, 8, 1000000000);
        array_ = std::make_unique<SystolicArray<float>>(config_);
    }
    
    SystolicArrayConfig config_;
    std::unique_ptr<SystolicArray<float>> array_;
};

TEST_F(SystolicArrayTest, Configuration) {
    EXPECT_EQ(config_.rows, 8);
    EXPECT_EQ(config_.cols, 8);
    EXPECT_EQ(config_.totalPes(), 64);
    EXPECT_GT(config_.peakTflops(), 0);
}

TEST_F(SystolicArrayTest, MatrixMultiply) {
    // 2x2 identity times 2x2 matrix = same matrix
    std::vector<std::vector<float>> a = {{1, 0}, {0, 1}};
    std::vector<std::vector<float>> b = {{1, 2}, {3, 4}};
    
    auto result = array_->multiply(a, b);
    
    EXPECT_GT(result.cycles, 0);
    EXPECT_GT(result.latency_us, 0);
    EXPECT_GT(result.throughput_tflops, 0);
}

TEST_F(SystolicArrayTest, Reset) {
    array_->reset();
    // Should not throw
}

TEST_F(SystolicArrayTest, EmptyMatrix) {
    std::vector<std::vector<float>> empty;
    std::vector<std::vector<float>> b = {{1, 2}, {3, 4}};
    
    auto result = array_->multiply(empty, b);
    EXPECT_EQ(result.cycles, 0);
}

TEST_F(SystolicArrayTest, PrecisionConfig) {
    PrecisionConfig config(DataType::FP16);
    EXPECT_EQ(config.dtype, DataType::FP16);
    
    config.dtype = DataType::INT8;
    EXPECT_EQ(config.dtype, DataType::INT8);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
