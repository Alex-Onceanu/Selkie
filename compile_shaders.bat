@echo off
setlocal

REM --- Locate Vulkan SDK ---
if "%VULKAN_SDK%"=="" (
    echo VULKAN_SDK not set!
    exit /b 1
)

set GLSLC="%VULKAN_SDK%\Bin\glslc.exe"

REM --- Paths ---
set SHADER_SRC=shaders
set SHADER_OUT=shaders\out

if not exist %SHADER_OUT% (
    mkdir %SHADER_OUT%
)

echo Compiling shaders...

for %%f in (%SHADER_SRC%\*) do (
    echo %%~nxf
    %GLSLC% --target-env=vulkan1.2 %%f -o %SHADER_OUT%\%%~nxf.spv
)

echo Done.
endlocal