cmake_minimum_required(VERSION 3.16)
project(program CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "C:/Users/YURI/.utils/llvm-mingw/bin/clang++.exe")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target x86_64-w64-windows-gnu -Wall -fopenmp")

# Includes
include_directories(inc)
include_directories("C:/Users/YURI/.utils/llvm-mingw/lib/gcc/x86_64-w64-mingw32/13.2.0")

# Fontes
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp main.cpp)

# Executável
add_executable(program ${SOURCES})

# Saída na pasta lib/bin
set_target_properties(program PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/bin"
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP encontrado!")
    target_link_libraries(program PRIVATE OpenMP::OpenMP_CXX)
endif()
