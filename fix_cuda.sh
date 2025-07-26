#!/bin/bash
set -e

FILE="/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h"

# Backup
sudo cp "$FILE" "$FILE.bak"

# Apply patches
sudo sed -i 's/cospi(double x);/cospi(double x) noexcept (true);/g' "$FILE"
sudo sed -i 's/__cospi(double x);/__cospi(double x) noexcept (true);/g' "$FILE"
sudo sed -i 's/sinpi(double x);/sinpi(double x) noexcept (true);/g' "$FILE"
sudo sed -i 's/__sinpi(double x);/__sinpi(double x) noexcept (true);/g' "$FILE"
sudo sed -i 's/cospif(float x);/cospif(float x) noexcept (true);/g' "$FILE"
sudo sed -i 's/__cospif(float x);/__cospif(float x) noexcept (true);/g' "$FILE"
sudo sed -i 's/sinpif(float x);/sinpif(float x) noexcept (true);/g' "$FILE"
sudo sed -i 's/__sinpif(float x);/__sinpif(float x) noexcept (true);/g' "$FILE"

# Verify patches
echo "Verifying patches:"
grep -E "cospi|sinpi|cospif|sinpif" "$FILE" | grep "noexcept (true)" || echo "Patches not found."

# Build and verify
echo "Building llama-cpp-python..."
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 uv pip install llama-cpp-python --no-cache-dir
if [ $? -eq 0 ]; then
  echo "Build successful."
else
  echo "Build failed."
fi