cmake_minimum_required(VERSION 3.15)
project(EdenValidator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CRITICAL: Deterministic floating-point settings
if(MSVC)
    # Disable floating-point optimizations that break determinism
    add_compile_options(/fp:strict /fp:except)
    # Disable SSE/AVX that can cause non-determinism
    add_compile_options(/arch:SSE2)
else()
    add_compile_options(-ffp-contract=off -fno-fast-math -msse2 -mno-avx)
endif()

# Find Bullet Physics
find_package(Bullet CONFIG REQUIRED)

# Create executable
add_executable(eden_validator src/validator.cpp)

# Link Bullet libraries
target_link_libraries(eden_validator 
    PRIVATE
    ${BULLET_LIBRARIES}
)

target_include_directories(eden_validator 
    PRIVATE 
    ${BULLET_INCLUDE_DIRS}
)

# Warnings
if(MSVC)
    target_compile_options(eden_validator PRIVATE /W4)
else()
    target_compile_options(eden_validator PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Use O2 instead of O3 for better determinism
if(CMAKE_BUILD_TYPE MATCHES Release)
    if(MSVC)
        target_compile_options(eden_validator PRIVATE /O2)
    else()
        target_compile_options(eden_validator PRIVATE -O2)
    endif()
endif()
