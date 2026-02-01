cmake_minimum_required(VERSION 3.16)
project(chronos_particle_lenia)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)


add_library(glad src/glad.c)
target_include_directories(glad PUBLIC ${PROJECT_SOURCE_DIR}/include)


include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)


add_library(imgui
    include/imgui/imgui.cpp
    include/imgui/imgui_draw.cpp
    include/imgui/imgui_tables.cpp
    include/imgui/imgui_widgets.cpp
    include/imgui/imgui_impl_glfw.cpp
    include/imgui/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${PROJECT_SOURCE_DIR}/include)


add_library(chronos_core
    src/core/Buffer.cpp
    src/core/Shader.cpp
    src/core/ComputeShader.cpp
    src/core/RenderShader.cpp
)
target_include_directories(chronos_core PUBLIC ${PROJECT_SOURCE_DIR}/src)


add_executable(particle_lenia
    src/particle_lenia/main.cpp
)
target_link_libraries(particle_lenia
    chronos_core
    imgui
    glad
    glfw
    OpenGL::GL
    OpenMP::OpenMP_CXX
    m
)


add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    COMMENT "Copying shaders to build directory"
)
add_dependencies(particle_lenia copy_shaders)
