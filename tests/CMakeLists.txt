# Cortex Tests CMakeLists.txt

# Fetch or include a testing framework
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executable
add_executable(cortex_tests 
    test_systolic_array.cpp
    test_tpu.cpp
    test_quantization.cpp
    test_roofline.cpp
    test_benchmark.cpp
)

target_link_libraries(cortex_tests PRIVATE 
    cortex_lib 
    gtest_main 
    gmock
    Threads::Threads
)

# Enable testing
enable_testing()
add_test(NAME cortex_tests COMMAND cortex_tests)

# Copy test resources if needed
