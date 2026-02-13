#!/bin/bash

for file in ../shaders/*.{rgen,rmiss,rint,rchit,rahit}; do
  glslc --target-env=vulkan1.4 "$file" -o "../shaders/out/$(basename "$file").spv"
done
