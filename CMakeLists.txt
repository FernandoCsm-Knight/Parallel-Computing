cmake_minimum_required(VERSION 3.16)
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
