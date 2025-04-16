cmake_minimum_required(VERSION 3.16)
<<<<<<< HEAD
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
=======
project(ProjetoComMainNaRaiz CXX)

# Usa C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Força o uso de clang++ se desejar (pode omitir se já for default)
set(CMAKE_CXX_COMPILER clang++)

# Ativa OpenMP se disponível
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP encontrado!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Inclui os headers da pasta inc/
include_directories(inc)

# Procura todos os arquivos .cpp da raiz (ou adicione manualmente)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)

# Cria o executável a partir dos fontes
add_executable(${PROJECT_NAME} ${SOURCES})
>>>>>>> 0bf5636dad2ed4cc28e89e012cd1d9f2f680e65c
