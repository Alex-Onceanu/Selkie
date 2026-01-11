#!/bin/bash

for file in shaders/*.{rgen,rchit,rmiss}; do
  glslc "$file" -o "shaders/out/$(basename "$file").spv"
done