/**
 * @file main.cpp
 * @brief Cortex - Neural Network Hardware Accelerator Emulator
 * @author Cortex Development Team
 * @version 1.0.0
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <sstream>

// Simple command line parser (inline for single-file simplicity)
struct Options {
    bool help = false;
    bool version = false;
    bool demo = false;
    bool list = false;
    bool benchmark = false;
    bool models = false;
    bool all = false;
    std::string benchmark_model;
    std::string models_list;
    
    static Options parse(int argc, char** argv) {
        Options opts;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") opts.help = true;
            else if (arg == "-v" || arg == "--version") opts.version = true;
            else if (arg == "-d" || arg == "--demo") opts.demo = true;
            else if (arg == "-l" || arg == "--list") opts.list = true;
            else if (arg == "-a" || arg == "--all") opts.all = true;
            else if (arg == "-b" || arg == "--benchmark") {
                opts.benchmark = true;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    opts.benchmark_model = argv[++i];
                }
            }
            else if (arg == "-m" || arg == "--models") {
                opts.models = true;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    opts.models_list = argv[++i];
                }
            }
        }
        return opts;
    }
    
    void printHelp() const {
        std::cout << R"(
Cortex - Neural Network Hardware Accelerator Emulator

Usage: cortex [options]

Options:
  -h, --help         Show this help message
  -v, --version      Show version information
  -d, --demo         Run demos
  -l, --list         List available models
  -b, --benchmark M  Run benchmark for model M
  -m, --models L     Run benchmarks for comma-separated list L
  -a, --all          Run all benchmarks

Examples:
  cortex --demo                    # Run demos
  cortex --list                     # List models
  cortex --benchmark resnet50       # Benchmark ResNet50
  cortex --models resnet50,gpt2     # Benchmark multiple models
  cortex --all                      # Run all benchmarks

Available Models:
  Vision: resnet50, resnet101, resnet152, vit_b_16, vit_l_16,
          convnext_tiny, convnext_base
  NLP: bert_base, bert_large, gpt2, gpt3, llama_7b, llama_70b
  GenAI: sd_unet, sd_vae, sd_text_encoder
  Audio: wavenet
)";
    }
};

#include "core/systolic_array.hpp"
#include "core/tpu_architecture.hpp"
#include "simulation/memory_bandwidth.hpp"
#include "simulation/dataflow_engine.hpp"
#include "simulation/batch_norm_fusion.hpp"
#include "simulation/quantization.hpp"
#include "analysis/latency_throughput.hpp"
#include "analysis/roofline_model.hpp"
#include "models/benchmark_models.hpp"
#include "utils/benchmark_runner.hpp"

using namespace cortex;
using namespace cortex::benchmark;

/**
 * @brief Print banner
 */
void printBanner() {
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗ ███████╗███╗   ██╗ ██████╗ ██╗     ███████╗      ║
    ║  ██╔════╝ ██╔════╝████╗  ██║██╔═══██╗██║     ██╔════╝      ║
    ║  ██║  ███╗█████╗  ██╔██╗ ██║██║   ██║██║     █████╗        ║
    ║  ██║   ██║██╔══╝  ██║╚██╗██║██║   ██║██║     ██╔══╝        ║
    ║  ╚██████╔╝███████╗██║ ╚████║╚██████╔╝███████╗███████╗      ║
    ║   ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝      ║
    ║                                                           ║
    ║   Neural Network Hardware Accelerator Emulator           ║
    ║   Version 1.0.0                                          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    )" << std::endl;
}

/**
 * @brief Print hardware information
 */
void printHardwareInfo() {
    std::cout << "\n=== Hardware Configuration ===\n";
    std::cout << "TPU Generation: V3\n";
    std::cout << "Peak Performance: 180 TFLOPS (FP16)\n";
    std::cout << "Peak Performance: 90 TFLOPS (FP32)\n";
    std::cout << "Peak Performance: 360 TOPS (INT8)\n";
    std::cout << "Memory Bandwidth: 900 GB/s (HBM)\n";
    std::cout << "Systolic Array: 128x128\n";
    std::cout << "Memory Capacity: 8 GB\n";
}

/**
 * @brief Demo systolic array
 */
void demoSystolicArray() {
    std::cout << "\n=== Systolic Array Demo ===\n";
    
    core::SystolicArrayConfig config(64, 64, 1000000000);
    core::SystolicArray<float> array(config);
    
    // Create test matrices
    std::vector<std::vector<float>> a = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
        {13.0f, 14.0f, 15.0f, 16.0f}
    };
    
    std::vector<std::vector<float>> b = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
    
    auto result = array.multiply(a, b);
    
    std::cout << "Matrix Size: 4x4\n";
    std::cout << "Cycles: " << result.cycles << "\n";
    std::cout << "Latency: " << std::fixed << std::setprecision(2) 
              << result.latency_us << " us\n";
    std::cout << "Throughput: " << result.throughput_tflops << " TFLOPS\n";
    std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOPs/byte\n";
}

/**
 * @brief Demo TPU architecture
 */
void demoTPU() {
    std::cout << "\n=== TPU Architecture Demo ===\n";
    
    auto tpu = std::make_unique<core::TPUArchitecture>(core::TPUGeneration::TPU_V3);
    
    std::cout << "Generation: TPU V3\n";
    std::cout << "Peak TFLOPS: " << std::fixed << std::setprecision(2) 
              << tpu->getPeakTflops() << " TFLOPS\n";
    std::cout << "Clock Frequency: " << tpu->getClockFrequency() / 1e9 << " GHz\n";
    std::cout << "HBM Bandwidth: " << tpu->getMemoryConfig().hbm_bandwidth_gbps << " GB/s\n";
}

/**
 * @brief Demo quantization
 */
void demoQuantization() {
    std::cout << "\n=== Quantization Demo ===\n";
    
    simulation::QuantizationCalibrator calibrator;
    
    // Generate some sample data
    std::vector<float> data(1000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : data) {
        val = dist(gen) * 10.0f;
    }
    
    calibrator.collectStats(data);
    auto params = calibrator.computeParams(simulation::Precision::INT8);
    
    std::cout << "Calibration samples: " << params.calibration_samples << "\n";
    std::cout << "Quantization scale: " << std::fixed << std::setprecision(6) 
              << params.params.scale << "\n";
    std::cout << "Data range: [" << params.observed_min << ", " 
              << params.observed_max << "]\n";
    
    // Test quantization
    simulation::Quantizer quantizer(params.params);
    auto quantized = quantizer.quantize(data);
    auto dequantized = quantizer.dequantize(quantized);
    
    // Compute error
    double mse = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        double err = data[i] - dequantized[i];
        mse += err * err;
    }
    mse /= data.size();
    
    std::cout << "Reconstruction MSE: " << std::scientific << std::setprecision(2) 
              << mse << "\n";
}

/**
 * @brief Demo batch norm fusion
 */
void demoBatchNormFusion() {
    std::cout << "\n=== Batch Norm Fusion Demo ===\n";
    
    simulation::BatchNormFusion fusion;
    
    // Create test conv weights (out_channels=64, in_channels=32, kernel=3x3)
    std::vector<float> conv_weight(64 * 32 * 3 * 3, 1.0f);
    
    // Create batch norm parameters
    simulation::BatchNormParams bn;
    bn.num_channels = 64;
    bn.gamma.resize(64, 1.0f);
    bn.beta.resize(64, 0.0f);
    bn.mean.resize(64, 0.0f);
    bn.variance.resize(64, 1.0f);
    bn.epsilon = 1e-5f;
    
    std::array<int, 4> conv_shape = {64, 32, 3, 3};
    
    auto result = fusion.fuseConvBN(conv_weight, {}, bn, conv_shape);
    
    std::cout << "Original memory: " << conv_weight.size() * sizeof(float) << " bytes\n";
    std::cout << "Fused memory: " << result.fused_weights.size() * sizeof(float) << " bytes\n";
    std::cout << "Memory saved: " << result.memory_saved_bytes << " bytes\n";
    std::cout << "Cycles saved: " << result.cycles_saved << "\n";
    std::cout << "Latency improvement: " << std::fixed << std::setprecision(2) 
              << result.latency_improvement_percent << "%\n";
}

/**
 * @brief Demo roofline analysis
 */
void demoRoofline() {
    std::cout << "\n=== Roofline Model Analysis Demo ===\n";
    
    analysis::PeakSpecifications peak;
    peak.fp16_tflops = 180;
    peak.hbm_bandwidth_gbps = 900;
    
    analysis::RooflineModel roofline(peak);
    
    std::cout << "Peak Performance: " << peak.fp16_tflops << " TFLOPS\n";
    std::cout << "Memory Bandwidth: " << peak.hbm_bandwidth_gbps << " GB/s\n";
    std::cout << "Knee Point (AI): " << std::fixed << std::setprecision(2) 
              << peak.fp16_tflops * 1000 / peak.hbm_bandwidth_gbps << " FLOPs/byte\n\n";
    
    // Analyze different kernel types
    std::vector<std::tuple<std::string, uint64_t, uint64_t>> kernels = {
        {"Conv 3x3", 3'900'000'000ULL, 100'000'000ULL},
        {"Conv 1x1", 2'100'000'000ULL, 50'000'000ULL},
        {"GEMM", 1'000'000'000ULL, 10'000'000ULL},
        {"Attention", 500'000'000'000ULL, 500'000'000ULL},
        {"LayerNorm", 10'000'000ULL, 5'000'000ULL}
    };
    
    std::cout << std::setw(15) << std::left << "Kernel"
              << std::setw(15) << "AI (FLOPs/B)"
              << std::setw(15) << "Bound"
              << std::setw(15) << "Efficiency"
              << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& [name, flops, bytes] : kernels) {
        auto analysis = roofline.analyzeKernel(name, flops, bytes);
        double efficiency = roofline.calculateEfficiency(analysis.achieved_tflops, "fp16");
        
        std::cout << std::setw(15) << std::left << name
                  << std::setw(15) << std::fixed << std::setprecision(2) << analysis.arithmetic_intensity
                  << std::setw(15) << analysis.bottleneck
                  << std::setw(15) << std::setprecision(1) << efficiency << "%"
                  << "\n";
    }
}

/**
 * @brief Run model benchmarks
 */
void runBenchmarks(const std::vector<std::string>& models) {
    std::cout << "\n=== Running Benchmarks ===\n\n";
    
    HardwareConfig hw;
    hw.name = "TPU_V3";
    hw.peak_specs.fp16_tflops = 180;
    hw.peak_specs.hbm_bandwidth_gbps = 900;
    
    BenchmarkRunner runner(hw);
    
    std::vector<BenchmarkResult> results;
    for (const auto& model : models) {
        std::cout << "Benchmarking " << model << "...\n";
        results.push_back(runner.run(model, 1, 512, 100, true));
    }
    
    std::cout << "\n" << runner.generateReport(results);
}

/**
 * @brief Print available models
 */
void printAvailableModels() {
    std::cout << "\n=== Available Models ===\n";
    
    std::vector<std::string> models = models::ModelFactory::availableModels();
    
    std::cout << "Vision Models:\n";
    std::cout << "  - resnet50, resnet101, resnet152\n";
    std::cout << "  - vit_b_16, vit_l_16\n";
    std::cout << "  - convnext_tiny, convnext_base\n\n";
    
    std::cout << "Transformer Models:\n";
    std::cout << "  - bert_base, bert_large\n";
    std::cout << "  - gpt2, gpt3, llama_7b, llama_70b\n\n";
    
    std::cout << "Generative Models:\n";
    std::cout << "  - sd_unet, sd_vae, sd_text_encoder\n\n";
    
    std::cout << "Audio Models:\n";
    std::cout << "  - wavenet\n";
}

int main(int argc, char* argv[]) {
    printBanner();
    
    auto opts = Options::parse(argc, argv);
    
    if (opts.help) {
        opts.printHelp();
        return 0;
    }
    
    if (opts.version) {
        std::cout << "Cortex v1.0.0\n";
        return 0;
    }
    
    if (opts.list) {
        printAvailableModels();
        return 0;
    }
    
    printHardwareInfo();
    
    if (opts.demo) {
        demoSystolicArray();
        demoTPU();
        demoQuantization();
        demoBatchNormFusion();
        demoRoofline();
    }
    
    if (opts.all) {
        std::vector<std::string> models = {
            "resnet50", "bert_base", "gpt2", "vit_b_16"
        };
        runBenchmarks(models);
    }
    
    if (opts.benchmark && !opts.benchmark_model.empty()) {
        runBenchmarks({opts.benchmark_model});
    }
    
    if (opts.models && !opts.models_list.empty()) {
        std::vector<std::string> models;
        std::stringstream ss(opts.models_list);
        std::string token;
        while (std::getline(ss, token, ',')) {
            models.push_back(token);
        }
        runBenchmarks(models);
    }
    
    // Default: run demos if no specific action
    if (!opts.demo && !opts.benchmark && !opts.models && !opts.all && !opts.list) {
        std::cout << "\nRunning demos...\n\n";
        demoSystolicArray();
        demoTPU();
        demoQuantization();
        demoBatchNormFusion();
        demoRoofline();
        
        std::cout << "\n\nTo run benchmarks, use:\n";
        std::cout << "  ./cortex --demo       # Run demos\n";
        std::cout << "  ./cortex --benchmark resnet50\n";
        std::cout << "  ./cortex --all       # Run all benchmarks\n";
        std::cout << "  ./cortex --help      # Show all options\n";
    }
    
    std::cout << "\n=== Cortex Simulation Complete ===\n";
    
    return 0;
}
