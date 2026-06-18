#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
cp _brstboost*.so ../
echo "Build done: $(ls ../_brstboost*.so)"
