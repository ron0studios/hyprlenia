cmake_minimum_required(VERSION 3.25)
project(chronos VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Fetch GLAD (OpenGL 4.3 for compute shaders)
include(FetchContent)
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG glad2
    SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(glad)
glad_add_library(glad_gl_core_43 REPRODUCIBLE API gl:core=4.3)

# Fetch ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)
FetchContent_MakeAvailable(imgui)

# ImGui source files
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# Main executable - 2D Flow Lenia
add_executable(chronos
    src/main.cpp
    src/ChronosApp2D.cpp
    src/ChronosApp2D.h
    src/FlowLenia2D.cpp
    src/FlowLenia2D.h
    src/BloomEffect.cpp
    src/BloomEffect.h
    ${IMGUI_SOURCES}
)

target_include_directories(chronos PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    src
)

target_link_libraries(chronos PRIVATE
    OpenGL::GL
    glfw
    glad_gl_core_43
    glm::glm
)

# Copy shaders to build directory
add_custom_command(TARGET chronos POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/shaders" "$<TARGET_FILE_DIR:chronos>/shaders"
)
