# Cortex source directory CMakeLists.txt

# Create Cortex library
set(CORTEX_SOURCES
    # Core modules
    core/systolic_array.hpp
    core/tpu_architecture.hpp
    
    # Simulation modules
    simulation/memory_bandwidth.hpp
    simulation/dataflow_engine.hpp
    simulation/batch_norm_fusion.hpp
    simulation/quantization.hpp
    
    # Analysis modules
    analysis/latency_throughput.hpp
    analysis/roofline_model.hpp
    
    # Models
    models/benchmark_models.hpp
    
    # Utils
    utils/benchmark_runner.hpp
)

# Create header-only library
add_library(cortex_lib INTERFACE)
target_include_directories(cortex_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_sources(cortex_lib INTERFACE 
    $<TARGET_PROPERTY:cortex_lib,INTERFACE_SOURCES>
    ${CORTEX_SOURCES}
)

# Create main executable
add_executable(cortex main.cpp)
target_link_libraries(cortex PRIVATE cortex_lib Threads::Threads)

# Install rules
install(TARGETS cortex cortex_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    DESTINATION include/cortex
    FILES_MATCHING_PATTERN "*.hpp"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/cortex/src
    FILES_MATCHING_PATTERN "*.hpp"
)

# Add tests subdirectory
if(CORTEX_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests ${CMAKE_BINARY_DIR}/tests)
endif()

# Add benchmarks subdirectory
if(CORTEX_BUILD_BENCHMARKS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks ${CMAKE_BINARY_DIR}/benchmarks)
endif()

# Print source files
message(STATUS "Cortex source files:")
foreach(src ${CORTEX_SOURCES})
    message(STATUS "  ${src}")
endforeach()
