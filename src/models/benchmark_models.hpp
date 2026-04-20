/**
 * @file benchmark_models.hpp
 * @brief Standard Neural Network Benchmark Models
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "core/tpu_architecture.hpp"
#include "analysis/latency_throughput.hpp"
#include "simulation/dataflow_engine.hpp"

namespace cortex {
namespace models {

/**
 * @brief Model architecture descriptor
 */
struct ModelConfig {
    std::string name;
    std::string architecture;
    uint32_t batch_size;
    std::vector<uint32_t> input_shape;
    std::vector<uint32_t> output_shape;
    uint64_t total_params;
    uint64_t total_flops;
    uint64_t activations_memory_bytes;
    uint64_t weights_memory_bytes;
    
    ModelConfig() : batch_size(1), total_params(0), total_flops(0),
                    activations_memory_bytes(0), weights_memory_bytes(0) {}
};

/**
 * @brief ResNet model family
 */
class ResNetModel {
public:
    static ModelConfig ResNet50(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "ResNet50";
        config.architecture = "ResNet";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 3, 224, 224};
        config.output_shape = {batch_size, 1000};
        config.total_params = 25'600'000;
        config.total_flops = 3'900'000'000ULL;
        config.activations_memory_bytes = 11'000'000;
        config.weights_memory_bytes = 98'000'000;
        return config;
    }
    
    static ModelConfig ResNet101(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "ResNet101";
        config.architecture = "ResNet";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 3, 224, 224};
        config.output_shape = {batch_size, 1000};
        config.total_params = 44'500'000;
        config.total_flops = 7'600'000'000ULL;
        config.activations_memory_bytes = 16'000'000;
        config.weights_memory_bytes = 170'000'000;
        return config;
    }
    
    static ModelConfig ResNet152(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "ResNet152";
        config.architecture = "ResNet";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 3, 224, 224};
        config.output_shape = {batch_size, 1000};
        config.total_params = 60'200'000;
        config.total_flops = 11'200'000'000ULL;
        config.activations_memory_bytes = 22'000'000;
        config.weights_memory_bytes = 230'000'000;
        return config;
    }
};

/**
 * @brief Transformer/LLM model family
 */
class TransformerModel {
public:
    static ModelConfig BERT_BASE(uint32_t batch_size = 1, uint32_t seq_len = 512) {
        ModelConfig config;
        config.name = "BERT_BASE";
        config.architecture = "Transformer";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, seq_len, 768};
        config.output_shape = {batch_size, seq_len, 30522};
        config.total_params = 110'000'000;
        config.total_flops = 33'000'000'000ULL;
        config.activations_memory_bytes = 256'000'000;
        config.weights_memory_bytes = 420'000'000;
        return config;
    }
    
    static ModelConfig BERT_LARGE(uint32_t batch_size = 1, uint32_t seq_len = 512) {
        ModelConfig config;
        config.name = "BERT_LARGE";
        config.architecture = "Transformer";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, seq_len, 1024};
        config.output_shape = {batch_size, seq_len, 30522};
        config.total_params = 340'000'000;
        config.total_flops = 113'000'000'000ULL;
        config.activations_memory_bytes = 512'000'000;
        config.weights_memory_bytes = 1'300'000'000;
        return config;
    }
    
    static ModelConfig GPT2(uint32_t batch_size = 1, uint32_t seq_len = 1024) {
        ModelConfig config;
        config.name = "GPT2";
        config.architecture = "Transformer";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, seq_len, 768};
        config.output_shape = {batch_size, seq_len, 50257};
        config.total_params = 150'000'000;
        config.total_flops = 48'000'000'000ULL;
        config.activations_memory_bytes = 384'000'000;
        config.weights_memory_bytes = 570'000'000;
        return config;
    }
    
    static ModelConfig GPT3_175B(uint32_t batch_size = 1, uint32_t seq_len = 2048) {
        ModelConfig config;
        config.name = "GPT3_175B";
        config.architecture = "Transformer";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, seq_len, 12288};
        config.output_shape = {batch_size, seq_len, 50257};
        config.total_params = 175'000'000'000ULL;
        config.total_flops = 1'200'000'000'000ULL;
        config.activations_memory_bytes = 90'000'000'000ULL;
        config.weights_memory_bytes = 700'000'000'000ULL;
        return config;
    }
    
    static ModelConfig LLaMA_7B(uint32_t batch_size = 1, uint32_t seq_len = 2048) {
        ModelConfig config;
        config.name = "LLaMA_7B";
        config.architecture = "Transformer";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, seq_len, 4096};
        config.output_shape = {batch_size, seq_len, 32000};
        config.total_params = 6'700'000'000ULL;
        config.total_flops = 55'000'000'000'000ULL;
        config.activations_memory_bytes = 8'000'000'000ULL;
        config.weights_memory_bytes = 26'000'000'000ULL;
        return config;
    }
    
    static ModelConfig LLaMA_70B(uint32_t batch_size = 1, uint32_t seq_len = 2048) {
        ModelConfig config;
        config.name = "LLaMA_70B";
        config.architecture = "Transformer";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, seq_len, 8192};
        config.output_shape = {batch_size, seq_len, 32000};
        config.total_params = 68'000'000'000ULL;
        config.total_flops = 560'000'000'000'000ULL;
        config.activations_memory_bytes = 80'000'000'000ULL;
        config.weights_memory_bytes = 272'000'000'000ULL;
        return config;
    }
};

/**
 * @brief Vision Transformer model
 */
class ViTModel {
public:
    static ModelConfig ViT_B_16(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "ViT_B_16";
        config.architecture = "ViT";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 3, 224, 224};
        config.output_shape = {batch_size, 1000};
        config.total_params = 86'000'000;
        config.total_flops = 17'600'000'000ULL;
        config.activations_memory_bytes = 45'000'000;
        config.weights_memory_bytes = 330'000'000;
        return config;
    }
    
    static ModelConfig ViT_L_16(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "ViT_L_16";
        config.architecture = "ViT";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 3, 224, 224};
        config.output_shape = {batch_size, 1000};
        config.total_params = 304'000'000;
        config.total_flops = 59'000'000'000ULL;
        config.activations_memory_bytes = 150'000'000;
        config.weights_memory_bytes = 1'200'000'000;
        return config;
    }
};

/**
 * @brief ConvNeXt model family
 */
class ConvNeXtModel {
public:
    static ModelConfig ConvNeXt_Tiny(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "ConvNeXt_Tiny";
        config.architecture = "ConvNeXt";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 3, 224, 224};
        config.output_shape = {batch_size, 1000};
        config.total_params = 28'000'000;
        config.total_flops = 4'500'000'000ULL;
        config.activations_memory_bytes = 15'000'000;
        config.weights_memory_bytes = 107'000'000;
        return config;
    }
    
    static ModelConfig ConvNeXt_Base(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "ConvNeXt_Base";
        config.architecture = "ConvNeXt";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 3, 224, 224};
        config.output_shape = {batch_size, 1000};
        config.total_params = 88'000'000;
        config.total_flops = 15'000'000'000ULL;
        config.activations_memory_bytes = 45'000'000;
        config.weights_memory_bytes = 337'000'000;
        return config;
    }
};

/**
 * @brief Stable Diffusion model components
 */
class StableDiffusionModel {
public:
    static ModelConfig UNet(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "SD_UNet";
        config.architecture = "UNet";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 4, 64, 64};
        config.output_shape = {batch_size, 4, 64, 64};
        config.total_params = 865'000'000;
        config.total_flops = 28'000'000'000'000ULL;
        config.activations_memory_bytes = 4'000'000'000ULL;
        config.weights_memory_bytes = 3'500'000'000ULL;
        return config;
    }
    
    static ModelConfig VAE(uint32_t batch_size = 1) {
        ModelConfig config;
        config.name = "SD_VAE";
        config.architecture = "VAE";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, 512, 512, 3};
        config.output_shape = {batch_size, 512, 512, 3};
        config.total_params = 84'000'000;
        config.total_flops = 75'000'000'000ULL;
        config.activations_memory_bytes = 500'000'000;
        config.weights_memory_bytes = 320'000'000;
        return config;
    }
    
    static ModelConfig TextEncoder(uint32_t batch_size = 1, uint32_t seq_len = 77) {
        ModelConfig config;
        config.name = "SD_TextEncoder";
        config.architecture = "Transformer";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, seq_len};
        config.output_shape = {batch_size, seq_len, 768};
        config.total_params = 123'000'000;
        config.total_flops = 9'000'000'000ULL;
        config.activations_memory_bytes = 50'000'000;
        config.weights_memory_bytes = 470'000'000;
        return config;
    }
};

/**
 * @brief WaveNet model
 */
class WaveNetModel {
public:
    static ModelConfig WaveNet(uint32_t batch_size = 1, uint32_t audio_length = 16000) {
        ModelConfig config;
        config.name = "WaveNet";
        config.architecture = "WaveNet";
        config.batch_size = batch_size;
        config.input_shape = {batch_size, audio_length};
        config.output_shape = {batch_size, audio_length, 256};
        config.total_params = 24'000'000;
        config.total_flops = 38'000'000'000ULL;
        config.activations_memory_bytes = 800'000'000;
        config.weights_memory_bytes = 92'000'000;
        return config;
    }
};

/**
 * @brief Model factory
 */
class ModelFactory {
public:
    static ModelConfig create(const std::string& name, 
                              uint32_t batch_size = 1,
                              uint32_t seq_len = 512) {
        
        // Vision models
        if (name == "resnet50" || name == "ResNet50") {
            return ResNetModel::ResNet50(batch_size);
        }
        if (name == "resnet101" || name == "ResNet101") {
            return ResNetModel::ResNet101(batch_size);
        }
        if (name == "resnet152" || name == "ResNet152") {
            return ResNetModel::ResNet152(batch_size);
        }
        if (name == "vit_b_16" || name == "ViT_B_16") {
            return ViTModel::ViT_B_16(batch_size);
        }
        if (name == "vit_l_16" || name == "ViT_L_16") {
            return ViTModel::ViT_L_16(batch_size);
        }
        if (name == "convnext_tiny" || name == "ConvNeXt_Tiny") {
            return ConvNeXtModel::ConvNeXt_Tiny(batch_size);
        }
        if (name == "convnext_base" || name == "ConvNeXt_Base") {
            return ConvNeXtModel::ConvNeXt_Base(batch_size);
        }
        
        // Transformer models
        if (name == "bert_base" || name == "BERT_BASE") {
            return TransformerModel::BERT_BASE(batch_size, seq_len);
        }
        if (name == "bert_large" || name == "BERT_LARGE") {
            return TransformerModel::BERT_LARGE(batch_size, seq_len);
        }
        if (name == "gpt2" || name == "GPT2") {
            return TransformerModel::GPT2(batch_size, seq_len);
        }
        if (name == "gpt3" || name == "GPT3_175B") {
            return TransformerModel::GPT3_175B(batch_size, seq_len);
        }
        if (name == "llama_7b" || name == "LLaMA_7B") {
            return TransformerModel::LLaMA_7B(batch_size, seq_len);
        }
        if (name == "llama_70b" || name == "LLaMA_70B") {
            return TransformerModel::LLaMA_70B(batch_size, seq_len);
        }
        
        // Stable Diffusion
        if (name == "sd_unet" || name == "SD_UNet") {
            return StableDiffusionModel::UNet(batch_size);
        }
        if (name == "sd_vae" || name == "SD_VAE") {
            return StableDiffusionModel::VAE(batch_size);
        }
        if (name == "sd_text_encoder") {
            return StableDiffusionModel::TextEncoder(batch_size, seq_len);
        }
        
        // Audio
        if (name == "wavenet" || name == "WaveNet") {
            return WaveNetModel::WaveNet(batch_size);
        }
        
        // Default
        return ResNetModel::ResNet50(batch_size);
    }
    
    static std::vector<std::string> availableModels() {
        return {
            "resnet50", "resnet101", "resnet152",
            "vit_b_16", "vit_l_16",
            "convnext_tiny", "convnext_base",
            "bert_base", "bert_large",
            "gpt2", "gpt3", "llama_7b", "llama_70b",
            "sd_unet", "sd_vae", "sd_text_encoder",
            "wavenet"
        };
    }
};

} // namespace models
} // namespace cortex
