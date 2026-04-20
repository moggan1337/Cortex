cmake_minimum_required(VERSION 3.14)
project(Cortex VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(CORTEX_BUILD_TESTS "Build tests" ON)
option(CORTEX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CORTEX_ENABLE_WERROR "Treat warnings as errors" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wno-unused-parameter")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

if(CORTEX_ENABLE_WERROR)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
    endif()
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(Threads REQUIRED)

# Add subdirectories
add_subdirectory(src)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Cortex Configuration ===")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build Tests: ${CORTEX_BUILD_TESTS}")
message(STATUS "Build Benchmarks: ${CORTEX_BUILD_BENCHMARKS}")
message(STATUS "=======================")
message(STATUS "")
