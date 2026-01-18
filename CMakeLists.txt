cmake_minimum_required(VERSION 3.16)
project(Selkie)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (WIN32)
    add_compile_definitions(
        WIN32_LEAN_AND_MEAN
    )
    add_compile_options(/permissive-)

    include(FetchContent)

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.9
    )

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(glfw)
else()
    add_custom_target(linux_glslc
        COMMAND bash ../compile_shaders.sh
    )
endif()

find_package(Vulkan REQUIRED)

add_executable(${PROJECT_NAME} window.cpp renderer.cpp main.cpp)

target_link_libraries(${PROJECT_NAME} glfw ${GLFW_LIBRARIES} Vulkan::Vulkan)

if(TARGET linux_glslc)
    add_dependencies(${PROJECT_NAME} linux_glslc)
endif()