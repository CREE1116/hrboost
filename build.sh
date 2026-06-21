#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Building ctypes library using Makefile..."
make clean
make
cp libhrboost.dylib python/hrboost/ 2>/dev/null || true
cp libhrboost.so python/hrboost/ 2>/dev/null || true

# Optionally try building with CMake if cmake is installed
if command -v cmake &> /dev/null; then
    echo "cmake found, building pybind11 bindings..."
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(sysctl -n hw.logicalcpu)
    cp _brstboost*.so ../ || cp _brstboost*.dylib ../ || true
    cd ..
else
    echo "cmake not found, skipping pybind11 module build."
fi

echo "Build complete."
