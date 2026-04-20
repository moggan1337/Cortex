# Benchmarks CMakeLists.txt

add_executable(cortex_benchmarks
    bench_models.cpp
)

target_link_libraries(cortex_benchmarks PRIVATE 
    cortex_lib 
    Threads::Threads
)
