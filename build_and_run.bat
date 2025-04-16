@echo off
echo ===============================
echo Compilando o projeto com CMake
echo ===============================
cmake -G "MinGW Makefiles" -B build
cmake --build build

IF EXIST build\lib\bin\program.exe (
    echo.
    echo ===============================
    echo Executando o programa:
    echo ===============================
    build\lib\bin\program.exe
) ELSE (
    echo.
    echo ❌ ERRO: Executável não encontrado!
)
pause
