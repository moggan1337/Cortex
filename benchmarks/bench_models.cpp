/**
 * @file bench_models.cpp
 * @brief Model Benchmark Suite
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include "models/benchmark_models.hpp"
#include "utils/benchmark_runner.hpp"

using namespace cortex::benchmark;

int main() {
    std::cout << "=== Cortex Model Benchmarks ===\n\n";
    
    HardwareConfig hw;
    hw.name = "TPU_V3";
    hw.peak_specs.fp16_tflops = 180;
    hw.peak_specs.hbm_bandwidth_gbps = 900;
    
    BenchmarkRunner runner(hw);
    
    std::vector<std::string> models = {
        "resnet50", "resnet101", "vit_b_16",
        "bert_base", "gpt2", "convnext_base"
    };
    
    std::vector<BenchmarkResult> results;
    for (const auto& model : models) {
        auto result = runner.run(model, 1, 512, 100, true);
        results.push_back(result);
    }
    
    std::cout << runner.generateReport(results);
    
    return 0;
}
