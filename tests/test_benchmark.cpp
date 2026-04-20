/**
 * @file test_benchmark.cpp
 * @brief Tests for Benchmark Models and Runner
 */

#include <gtest/gtest.h>
#include "models/benchmark_models.hpp"
#include "utils/benchmark_runner.hpp"

using namespace cortex::models;
using namespace cortex::benchmark;

class BenchmarkTest : public ::testing::Test {};

TEST_F(BenchmarkTest, ResNetModels) {
    auto config = ModelFactory::create("resnet50");
    EXPECT_EQ(config.name, "ResNet50");
    EXPECT_EQ(config.architecture, "ResNet");
    EXPECT_GT(config.total_params, 0);
    EXPECT_GT(config.total_flops, 0);
    
    config = ModelFactory::create("resnet101");
    EXPECT_EQ(config.name, "ResNet101");
    EXPECT_GT(config.total_flops, ModelFactory::create("resnet50").total_flops);
}

TEST_F(BenchmarkTest, BERTModels) {
    auto config = ModelFactory::create("bert_base", 1, 512);
    EXPECT_EQ(config.name, "BERT_BASE");
    EXPECT_EQ(config.architecture, "Transformer");
    EXPECT_EQ(config.sequence_length, 512);
}

TEST_F(BenchmarkTest, GPTModels) {
    auto config = ModelFactory::create("gpt2");
    EXPECT_EQ(config.name, "GPT2");
    EXPECT_GT(config.total_params, 0);
}

TEST_F(BenchmarkTest, LLaMAModels) {
    auto config = ModelFactory::create("llama_7b");
    EXPECT_EQ(config.name, "LLaMA_7B");
    EXPECT_GT(config.total_params, 6'000'000'000ULL);
    
    config = ModelFactory::create("llama_70b");
    EXPECT_EQ(config.name, "LLaMA_70B");
    EXPECT_GT(config.total_params, 60'000'000'000ULL);
}

TEST_F(BenchmarkTest, ViTModels) {
    auto config = ModelFactory::create("vit_b_16");
    EXPECT_EQ(config.name, "ViT_B_16");
    EXPECT_EQ(config.architecture, "ViT");
}

TEST_F(BenchmarkTest, ConvNeXtModels) {
    auto config = ModelFactory::create("convnext_base");
    EXPECT_EQ(config.name, "ConvNeXt_BASE");
    EXPECT_EQ(config.architecture, "ConvNeXt");
}

TEST_F(BenchmarkTest, StableDiffusionModels) {
    auto config = ModelFactory::create("sd_unet");
    EXPECT_EQ(config.name, "SD_UNet");
    EXPECT_EQ(config.architecture, "UNet");
    
    config = ModelFactory::create("sd_vae");
    EXPECT_EQ(config.name, "SD_VAE");
}

TEST_F(BenchmarkTest, AvailableModels) {
    auto models = ModelFactory::availableModels();
    
    EXPECT_GE(models.size(), 10);
    EXPECT_NE(std::find(models.begin(), models.end(), "resnet50"), models.end());
    EXPECT_NE(std::find(models.begin(), models.end(), "gpt2"), models.end());
    EXPECT_NE(std::find(models.begin(), models.end(), "llama_7b"), models.end());
}

TEST_F(BenchmarkTest, BenchmarkRunner) {
    HardwareConfig hw;
    hw.peak_specs.fp16_tflops = 180;
    hw.peak_specs.hbm_bandwidth_gbps = 900;
    
    BenchmarkRunner runner(hw);
    auto result = runner.run("resnet50", 1, 512, 10, true);
    
    EXPECT_EQ(result.model_name, "resnet50");
    EXPECT_GT(result.latency_mean_us, 0);
    EXPECT_GT(result.throughput_samples_per_sec, 0);
}

TEST_F(BenchmarkTest, BenchmarkReport) {
    HardwareConfig hw;
    hw.peak_specs.fp16_tflops = 180;
    
    BenchmarkRunner runner(hw);
    std::vector<BenchmarkResult> results;
    results.push_back(runner.run("resnet50", 1, 512, 10, true));
    results.push_back(runner.run("gpt2", 1, 512, 10, true));
    
    auto report = runner.generateReport(results);
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("TFLOPS"), std::string::npos);
}

TEST_F(BenchmarkTest, QuickBenchmark) {
    auto result = quickBenchmark("resnet50", 1, true);
    
    EXPECT_EQ(result.model_name, "resnet50");
    EXPECT_GT(result.total_flops, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
