#!/bin/bash

for file in ../shaders/*.{rgen,rchit,rmiss,rint}; do
  glslc --target-env=vulkan1.2 "$file" -o "../shaders/out/$(basename "$file").spv"
done

for file in ../shaders/*.rahit; do
  glslc -fshader-stage=rahit --target-env=vulkan1.2 "$file" -o "../shaders/out/$(basename "$file").spv"
done