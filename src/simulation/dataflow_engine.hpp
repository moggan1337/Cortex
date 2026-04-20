/**
 * @file dataflow_engine.hpp
 * @brief Dataflow Execution Engine for Neural Network Accelerators
 * @author Cortex Development Team
 * @version 1.0.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <queue>
#include <variant>
#include <optional>
#include "core/systolic_array.hpp"
#include "core/tpu_architecture.hpp"

namespace cortex {
namespace simulation {

/**
 * @brief Dataflow execution strategies
 */
enum class DataflowStrategy {
    WEIGHT_STATIONARY,      // Weights stay in PE, inputs flow through
    OUTPUT_STATIONARY,      // Output accumulates in place
    INPUT_STATIONARY,       // Inputs stay in PE, weights flow through
    ROW_STATIONARY,         // Row of weights stays stationary
    ESCHER,                 // Google TPU style
    ShiDianNao,             // Local memory centric
    CUSTOM
};

/**
 * @brief Execution configuration
 */
struct ExecutionConfig {
    DataflowStrategy dataflow;
    uint32_t batch_size;
    uint32_t num_threads;
    bool enable_pipelining;
    bool enable_prefetch;
    bool enable_out_of_order;
    uint32_t max_queue_depth;
    
    ExecutionConfig() 
        : dataflow(DataflowStrategy::WEIGHT_STATIONARY),
          batch_size(1),
          num_threads(4),
          enable_pipelining(true),
          enable_prefetch(true),
          enable_out_of_order(false),
          max_queue_depth(32) {}
};

/**
 * @brief Operation type
 */
enum class OperationType {
    MATMUL,
    CONV2D,
    CONV3D,
    POOLING_AVG,
    POOLING_MAX,
    ACTIVATION,
    BATCH_NORM,
    DROPOUT,
    RESHAPE,
    TRANSPOSE,
    SOFTMAX,
    LSTM_CELL,
    GRU_CELL,
    ATTENTION,
    FUSED_LAYER_NORM
};

/**
 * @brief Tensor descriptor
 */
struct TensorDescriptor {
    std::string name;
    std::vector<int32_t> shape;
    core::DataType dtype;
    bool is_constant;
    bool is_buffer;
    uint64_t offset_bytes;
    
    TensorDescriptor() : dtype(core::DataType::FP32), 
                         is_constant(false), is_buffer(false), offset_bytes(0) {}
};

/**
 * @brief Operation descriptor
 */
struct OperationDescriptor {
    std::string name;
    OperationType type;
    std::vector<TensorDescriptor> inputs;
    std::vector<TensorDescriptor> outputs;
    std::vector<int> kernel_size;
    std::vector<int> strides;
    std::vector<int> padding;
    std::string fused_activation;
    
    // Quantization parameters
    float quant_scale;
    int32_t quant_zero_point;
    std::string quantization_scheme;
    
    OperationDescriptor() 
        : type(OperationType::MATMUL),
          quant_scale(1.0f),
          quant_zero_point(0),
          quantization_scheme("none") {}
};

/**
 * @brief Execution node in dataflow graph
 */
struct ExecutionNode {
    OperationDescriptor op;
    uint32_t id;
    std::vector<uint32_t> dependencies;
    std::vector<uint32_t> dependents;
    bool is_ready;
    bool is_executed;
    uint64_t scheduled_cycle;
    uint64_t completed_cycle;
    double estimated_latency_us;
    
    ExecutionNode() : id(0), is_ready(false), is_executed(false),
                      scheduled_cycle(0), completed_cycle(0), 
                      estimated_latency_us(0) {}
};

/**
 * @brief Dataflow graph
 */
class DataflowGraph {
public:
    DataflowGraph() : next_node_id_(0) {}
    
    /**
     * @brief Add operation to graph
     */
    uint32_t addOperation(const OperationDescriptor& op) {
        ExecutionNode node;
        node.id = next_node_id_++;
        node.op = op;
        nodes_.push_back(node);
        return node.id;
    }
    
    /**
     * @brief Add dependency between operations
     */
    void addDependency(uint32_t from, uint32_t to) {
        if (from < nodes_.size() && to < nodes_.size()) {
            nodes_[to].dependencies.push_back(from);
            nodes_[from].dependents.push_back(to);
        }
    }
    
    /**
     * @brief Topological sort
     */
    std::vector<uint32_t> topologicalSort() const {
        std::vector<uint32_t> result;
        std::vector<int> in_degree(nodes_.size(), 0);
        
        for (const auto& node : nodes_) {
            for (uint32_t dep : node.dependencies) {
                in_degree[node.id]++;
            }
        }
        
        std::queue<uint32_t> q;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (in_degree[i] == 0) {
                q.push(static_cast<uint32_t>(i));
            }
        }
        
        while (!q.empty()) {
            uint32_t node_id = q.front();
            q.pop();
            result.push_back(node_id);
            
            for (uint32_t dep : nodes_[node_id].dependents) {
                if (--in_degree[dep] == 0) {
                    q.push(dep);
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Get ready nodes (all dependencies satisfied)
     */
    std::vector<uint32_t> getReadyNodes(const std::vector<bool>& executed) const {
        std::vector<uint32_t> ready;
        for (const auto& node : nodes_) {
            if (executed[node.id]) continue;
            
            bool all_deps_done = true;
            for (uint32_t dep : node.dependencies) {
                if (!executed[dep]) {
                    all_deps_done = false;
                    break;
                }
            }
            if (all_deps_done) {
                ready.push_back(node.id);
            }
        }
        return ready;
    }
    
    const ExecutionNode& getNode(uint32_t id) const { return nodes_[id]; }
    size_t numNodes() const { return nodes_.size(); }

private:
    std::vector<ExecutionNode> nodes_;
    uint32_t next_node_id_;
};

/**
 * @brief Scheduling policy
 */
enum class SchedulingPolicy {
    FIFO,               // First in, first out
    LIFO,               // Last in, first out
    READY_FIRST,        // Ready nodes first
    CRITICAL_PATH,      // Longest path first
    MINIMUM_LATENCY,    // Minimize total latency
    LOOKAHEAD           // Look-ahead scheduling
};

/**
 * @brief Dataflow execution engine
 */
class DataflowEngine {
public:
    DataflowEngine(const ExecutionConfig& config = ExecutionConfig())
        : config_(config) {}
    
    /**
     * @brief Execute a dataflow graph
     */
    struct ExecutionResult {
        uint64_t total_cycles;
        uint64_t total_latency_us;
        double throughput_samples_per_sec;
        uint32_t num_stages;
        std::vector<std::pair<std::string, uint64_t>> stage_times;
        std::vector<uint32_t> execution_order;
        double pipeline_utilization;
        uint64_t pipeline_stalls;
    };
    
    ExecutionResult execute(const DataflowGraph& graph,
                           uint32_t num_iterations = 1) {
        ExecutionResult result;
        
        auto sorted_nodes = graph.topologicalSort();
        result.execution_order = sorted_nodes;
        result.num_stages = sorted_nodes.size();
        
        uint64_t current_cycle = 0;
        uint64_t max_latency = 0;
        
        for (uint32_t iter = 0; iter < num_iterations; ++iter) {
            std::vector<bool> executed(graph.numNodes(), false);
            std::vector<uint64_t> node_start_cycles(graph.numNodes(), 0);
            std::vector<uint64_t> node_end_cycles(graph.numNodes(), 0);
            
            // Simple sequential execution for now
            for (uint32_t node_id : sorted_nodes) {
                const auto& node = graph.getNode(node_id);
                
                // Wait for dependencies
                uint64_t start_cycle = current_cycle;
                for (uint32_t dep : node.dependencies) {
                    start_cycle = std::max(start_cycle, node_end_cycles[dep]);
                }
                
                node_start_cycles[node_id] = start_cycle;
                
                // Execute operation
                uint64_t latency = estimateLatency(node.op);
                node_end_cycles[node_id] = start_cycle + latency;
                
                // Track stage times
                bool found = false;
                for (auto& stage : result.stage_times) {
                    if (stage.first == node.op.name) {
                        stage.second += latency;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    result.stage_times.push_back({node.op.name, latency});
                }
            }
            
            current_cycle = *std::max_element(node_end_cycles.begin(), 
                                              node_end_cycles.end());
        }
        
        result.total_cycles = current_cycle;
        result.total_latency_us = current_cycle / 1000;  // Simplified
        
        if (num_iterations > 0) {
            result.throughput_samples_per_sec = 
                (num_iterations * 1e6) / result.total_latency_us;
        }
        
        // Calculate pipeline utilization
        uint64_t total_compute_cycles = 0;
        for (const auto& stage : result.stage_times) {
            total_compute_cycles += stage.second;
        }
        result.pipeline_utilization = 
            (double)total_compute_cycles / 
            (result.total_cycles * std::max(1u, result.num_stages));
        
        result.pipeline_stalls = 0;
        
        return result;
    }
    
    /**
     * @brief Schedule operations with specified policy
     */
    std::vector<uint32_t> schedule(const DataflowGraph& graph,
                                   SchedulingPolicy policy,
                                   uint32_t num_available_units = 1) {
        std::vector<uint32_t> schedule_order;
        std::vector<bool> executed(graph.numNodes(), false);
        std::vector<uint32_t> in_flight;
        
        while (schedule_order.size() < graph.numNodes()) {
            auto ready = graph.getReadyNodes(executed);
            
            if (ready.empty()) break;
            
            // Apply scheduling policy
            switch (policy) {
                case SchedulingPolicy::FIFO:
                    schedule_order.push_back(ready.front());
                    executed[ready.front()] = true;
                    break;
                    
                case SchedulingPolicy::CRITICAL_PATH: {
                    // Simple: schedule node with most dependents first
                    uint32_t best = ready[0];
                    size_t max_deps = graph.getNode(best).dependents.size();
                    for (uint32_t id : ready) {
                        size_t deps = graph.getNode(id).dependents.size();
                        if (deps > max_deps) {
                            max_deps = deps;
                            best = id;
                        }
                    }
                    schedule_order.push_back(best);
                    executed[best] = true;
                    break;
                }
                
                default:
                    schedule_order.push_back(ready[0]);
                    executed[ready[0]] = true;
            }
        }
        
        return schedule_order;
    }

private:
    uint64_t estimateLatency(const OperationDescriptor& op) {
        uint64_t base_cycles = 1000;
        
        // Estimate based on operation type
        switch (op.type) {
            case OperationType::MATMUL:
                base_cycles = 10000;
                break;
            case OperationType::CONV2D:
                base_cycles = 50000;
                break;
            case OperationType::ATTENTION:
                base_cycles = 20000;
                break;
            case OperationType::LSTM_CELL:
                base_cycles = 30000;
                break;
            default:
                base_cycles = 5000;
        }
        
        // Adjust based on tensor sizes
        size_t total_size = 1;
        for (const auto& input : op.inputs) {
            for (int dim : input.shape) {
                total_size *= (dim > 0) ? static_cast<size_t>(dim) : 1;
            }
        }
        
        return base_cycles + total_size / 1000;
    }
    
    ExecutionConfig config_;
};

/**
 * @brief Pipeline stage
 */
class PipelineStage {
public:
    PipelineStage(const std::string& name, uint32_t depth)
        : name_(name), depth_(depth), active_(false) {
        buffer_.reserve(depth);
    }
    
    /**
     * @brief Push item into pipeline
     */
    bool push(void* item) {
        if (buffer_.size() >= depth_) return false;
        buffer_.push_back(item);
        active_ = true;
        return true;
    }
    
    /**
     * @brief Pop item from pipeline
     */
    bool pop(void*& item) {
        if (buffer_.empty()) return false;
        item = buffer_.front();
        buffer_.erase(buffer_.begin());
        active_ = !buffer_.empty();
        return true;
    }
    
    /**
     * @brief Advance pipeline by one cycle
     */
    void tick() {
        if (buffer_.size() > 1) {
            std::rotate(buffer_.begin(), buffer_.begin() + 1, buffer_.end());
        }
    }
    
    bool isFull() const { return buffer_.size() >= depth_; }
    bool isEmpty() const { return buffer_.empty(); }
    bool isActive() const { return active_; }
    size_t occupancy() const { return buffer_.size(); }
    const std::string& name() const { return name_; }

private:
    std::string name_;
    uint32_t depth_;
    std::vector<void*> buffer_;
    bool active_;
};

/**
 * @brief Pipeline executor
 */
class PipelineExecutor {
public:
    PipelineExecutor() : cycle_(0), stalled_cycles_(0) {}
    
    void addStage(const std::string& name, uint32_t depth) {
        stages_.emplace_back(name, depth);
    }
    
    /**
     * @brief Execute pipeline for specified cycles
     */
    struct PipelineResult {
        uint64_t total_cycles;
        uint64_t stalled_cycles;
        double efficiency;
        std::vector<std::pair<std::string, size_t>> stage_occupancies;
    };
    
    PipelineResult execute(uint64_t num_items, uint64_t max_cycles = 100000) {
        PipelineResult result;
        
        while (items_processed_ < num_items && cycle_ < max_cycles) {
            bool stall = false;
            
            // Try to push new item
            if (!stages_.empty() && !stages_[0].isFull()) {
                stages_[0].push(nullptr);  // Simplified
                items_processed_++;
            } else if (!stages_.empty()) {
                stall = true;
            }
            
            // Advance all stages
            for (auto& stage : stages_) {
                stage.tick();
            }
            
            if (stall) stalled_cycles_++;
            cycle_++;
        }
        
        result.total_cycles = cycle_;
        result.stalled_cycles = stalled_cycles_;
        result.efficiency = (double)(cycle_ - stalled_cycles_) / cycle_;
        
        for (const auto& stage : stages_) {
            result.stage_occupancies.push_back({stage.name(), stage.occupancy()});
        }
        
        return result;
    }

private:
    std::vector<PipelineStage> stages_;
    uint64_t cycle_;
    uint64_t items_processed_;
    uint64_t stalled_cycles_;
};

} // namespace simulation
} // namespace cortex
