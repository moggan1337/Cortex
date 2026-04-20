/**
 * @file test_tpu.cpp
 * @brief Tests for TPU Architecture Module
 */

#include <gtest/gtest.h>
#include "core/tpu_architecture.hpp"

using namespace cortex::core;

class TPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        tpu_ = std::make_unique<TPUArchitecture>(TPUGeneration::TPU_V3);
    }
    
    std::unique_ptr<TPUArchitecture> tpu_;
};

TEST_F(TPUTest, TPUGeneration) {
    EXPECT_EQ(tpu_->getGeneration(), TPUGeneration::TPU_V3);
}

TEST_F(TPUTest, ClockFrequency) {
    auto freq = tpu_->getClockFrequency();
    EXPECT_GT(freq, 0);
    EXPECT_EQ(freq, 940000000);  // TPU V3 frequency
}

TEST_F(TPUTest, MemoryConfig) {
    const auto& mem = tpu_->getMemoryConfig();
    EXPECT_GT(mem.hbm_bandwidth_gbps, 0);
}

TEST_F(TPUTest, PeakTflops) {
    auto peak = tpu_->getPeakTflops();
    EXPECT_GT(peak, 0);
}

TEST_F(TPUTest, MXUConfig) {
    const auto& mxu = tpu_->getMXUConfig();
    EXPECT_EQ(mxu.rows, 128);
    EXPECT_EQ(mxu.cols, 128);
}

TEST_F(TPUTest, TensorShape) {
    TensorShape shape({1, 2, 3});
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 3);
    EXPECT_EQ(shape[3], 1);  // Default for out of bounds
    EXPECT_EQ(shape.size(), 6);
}

TEST_F(TPUTest, Tensor) {
    Tensor tensor({1, 2, 3}, DataType::FP16);
    EXPECT_EQ(tensor.numElements(), 6);
    EXPECT_EQ(tensor.dtype(), DataType::FP16);
}

TEST_F(TPUTest, UnifiedBufferAllocation) {
    UnifiedBuffer buffer(1024);
    
    uint64_t offset1 = buffer.allocate(256);
    EXPECT_NE(offset1, UINT64_MAX);
    EXPECT_GT(buffer.available(), 0);
    
    bool dealloc_ok = buffer.deallocate(offset1);
    EXPECT_TRUE(dealloc_ok);
}

TEST_F(TPUTest, UnifiedBufferOutOfMemory) {
    UnifiedBuffer buffer(100);
    
    uint64_t offset1 = buffer.allocate(50);
    uint64_t offset2 = buffer.allocate(50);
    uint64_t offset3 = buffer.allocate(1);  // Should fail
    
    EXPECT_NE(offset1, UINT64_MAX);
    EXPECT_NE(offset2, UINT64_MAX);
    EXPECT_EQ(offset3, UINT64_MAX);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
