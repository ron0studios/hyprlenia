cmake_minimum_required(VERSION 3.16)
project(chronos_particle_lenia)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)

# Headless rendering dependencies
pkg_check_modules(EGL REQUIRED egl)
pkg_check_modules(GBM REQUIRED gbm)

# FFmpeg for video encoding
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libswscale libavutil)

# GLAD library
add_library(glad src/glad.c)
target_include_directories(glad PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# ImGui library
add_library(imgui
    include/imgui/imgui.cpp
    include/imgui/imgui_draw.cpp
    include/imgui/imgui_tables.cpp
    include/imgui/imgui_widgets.cpp
    include/imgui/imgui_impl_glfw.cpp
    include/imgui/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Core library with OpenGL abstractions
add_library(chronos_core
    src/core/Buffer.cpp
    src/core/Shader.cpp
    src/core/ComputeShader.cpp
    src/core/RenderShader.cpp
)
target_include_directories(chronos_core PUBLIC ${PROJECT_SOURCE_DIR}/src)

# Headless rendering and video encoding library
add_library(chronos_headless
    src/headless/HeadlessRenderer.cpp
    src/headless/VideoEncoder.cpp
)
target_include_directories(chronos_headless PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${EGL_INCLUDE_DIRS}
    ${GBM_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)
target_link_libraries(chronos_headless
    ${EGL_LIBRARIES}
    ${GBM_LIBRARIES}
    ${FFMPEG_LIBRARIES}
)

# Main executable - Particle Lenia with Evolution
add_executable(particle_lenia
    src/particle_lenia/main.cpp
)
target_link_libraries(particle_lenia
    chronos_core
    chronos_headless
    imgui
    glad
    glfw
    OpenGL::GL
    OpenMP::OpenMP_CXX
    m
)

# Copy shaders to build directory
add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    COMMENT "Copying shaders to build directory"
)
add_dependencies(particle_lenia copy_shaders)
