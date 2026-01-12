#!/bin/bash

for file in shaders/*.{rgen,rchit,rmiss}; do
  glslc --target-env=vulkan1.2 "$file" -o "shaders/out/$(basename "$file").spv"
done