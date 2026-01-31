#include "VolumeRenderer.h"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <cstdio>

namespace {
// Vertex shader - renders a cube that we'll ray march through
const char* vertexShaderSource = R"(
#version 450 core
layout (location = 0) in vec3 aPos;

out vec3 vWorldPos;
out vec3 vLocalPos;

uniform mat4 view;
uniform mat4 projection;

void main() {
    vLocalPos = aPos;
    vWorldPos = aPos - 0.5; // Center the cube
    gl_Position = projection * view * vec4(vWorldPos, 1.0);
}
)";

// Fragment shader - ray marching through the volume
const char* fragmentShaderSource = R"(
#version 450 core
in vec3 vWorldPos;
in vec3 vLocalPos;

out vec4 FragColor;

uniform sampler3D volumeTex;
uniform vec3 cameraPos;
uniform float density;
uniform vec3 volumeSize;

// Color palette for Lenia
vec3 getLeniaColor(float value) {
    // Beautiful organic color palette
    vec3 c1 = vec3(0.02, 0.02, 0.05);  // Deep blue-black
    vec3 c2 = vec3(0.1, 0.2, 0.4);     // Dark blue
    vec3 c3 = vec3(0.2, 0.5, 0.6);     // Cyan
    vec3 c4 = vec3(0.4, 0.8, 0.5);     // Green
    vec3 c5 = vec3(0.9, 0.9, 0.3);     // Yellow
    vec3 c6 = vec3(1.0, 0.6, 0.2);     // Orange
    vec3 c7 = vec3(1.0, 0.3, 0.3);     // Red
    
    if (value < 0.15) return mix(c1, c2, value / 0.15);
    if (value < 0.3) return mix(c2, c3, (value - 0.15) / 0.15);
    if (value < 0.45) return mix(c3, c4, (value - 0.3) / 0.15);
    if (value < 0.6) return mix(c4, c5, (value - 0.45) / 0.15);
    if (value < 0.75) return mix(c5, c6, (value - 0.6) / 0.15);
    return mix(c6, c7, (value - 0.75) / 0.25);
}

void main() {
    // Ray direction from camera through this fragment
    vec3 rayDir = normalize(vWorldPos - cameraPos);
    vec3 rayOrigin = vWorldPos;
    
    // Ray marching parameters
    const int MAX_STEPS = 128;
    const float STEP_SIZE = 0.01;
    
    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;
    
    vec3 pos = rayOrigin;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        // Convert world pos to texture coordinates [0,1]
        vec3 texCoord = pos + 0.5;
        
        // Check if inside volume
        if (texCoord.x < 0.0 || texCoord.x > 1.0 ||
            texCoord.y < 0.0 || texCoord.y > 1.0 ||
            texCoord.z < 0.0 || texCoord.z > 1.0) {
            break;
        }
        
        // Sample the volume
        float value = texture(volumeTex, texCoord).r;
        
        if (value > 0.01) {
            // Get color based on value
            vec3 sampleColor = getLeniaColor(value);
            
            // Emission and absorption
            float sampleAlpha = value * density * STEP_SIZE * 10.0;
            sampleAlpha = min(sampleAlpha, 1.0);
            
            // Additive blending with emission
            accumColor += sampleColor * sampleAlpha * (1.0 - accumAlpha);
            accumAlpha += sampleAlpha * (1.0 - accumAlpha);
            
            if (accumAlpha > 0.95) break;
        }
        
        pos += rayDir * STEP_SIZE;
    }
    
    // Add a subtle glow
    accumColor *= 1.5;
    
    FragColor = vec4(accumColor, accumAlpha);
}
)";

// Cube vertices for the bounding box
float cubeVertices[] = {
    // Front face
    0, 0, 1,  1, 0, 1,  1, 1, 1,
    1, 1, 1,  0, 1, 1,  0, 0, 1,
    // Back face
    0, 0, 0,  0, 1, 0,  1, 1, 0,
    1, 1, 0,  1, 0, 0,  0, 0, 0,
    // Top face
    0, 1, 0,  0, 1, 1,  1, 1, 1,
    1, 1, 1,  1, 1, 0,  0, 1, 0,
    // Bottom face
    0, 0, 0,  1, 0, 0,  1, 0, 1,
    1, 0, 1,  0, 0, 1,  0, 0, 0,
    // Right face
    1, 0, 0,  1, 1, 0,  1, 1, 1,
    1, 1, 1,  1, 0, 1,  1, 0, 0,
    // Left face
    0, 0, 0,  0, 0, 1,  0, 1, 1,
    0, 1, 1,  0, 1, 0,  0, 0, 0,
};
} // namespace

VolumeRenderer::VolumeRenderer() = default;

VolumeRenderer::~VolumeRenderer() {
    shutdown();
}

void VolumeRenderer::init() {
    // Compile shaders
    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, nullptr);
    glCompileShader(vs);
    
    // Check for errors
    int success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vs, 512, nullptr, log);
        printf("Vertex shader error: %s\n", log);
    }
    
    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fs);
    
    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fs, 512, nullptr, log);
        printf("Fragment shader error: %s\n", log);
    }
    
    _shaderProgram = glCreateProgram();
    glAttachShader(_shaderProgram, vs);
    glAttachShader(_shaderProgram, fs);
    glLinkProgram(_shaderProgram);
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    
    // Get uniform locations
    _locView = glGetUniformLocation(_shaderProgram, "view");
    _locProjection = glGetUniformLocation(_shaderProgram, "projection");
    _locCameraPos = glGetUniformLocation(_shaderProgram, "cameraPos");
    _locVolumeTex = glGetUniformLocation(_shaderProgram, "volumeTex");
    _locDensity = glGetUniformLocation(_shaderProgram, "density");
    _locVolumeSize = glGetUniformLocation(_shaderProgram, "volumeSize");
    
    // Create VAO/VBO for cube
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
    
    // Create 3D texture
    glGenTextures(1, &_volumeTexture);
}

void VolumeRenderer::shutdown() {
    if (_shaderProgram) {
        glDeleteProgram(_shaderProgram);
        _shaderProgram = 0;
    }
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }
    if (_vbo) {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }
    if (_volumeTexture) {
        glDeleteTextures(1, &_volumeTexture);
        _volumeTexture = 0;
    }
}

void VolumeRenderer::updateTexture(const std::vector<float>& grid, int sizeX, int sizeY, int sizeZ) {
    if (grid.empty()) return;
    
    glBindTexture(GL_TEXTURE_3D, _volumeTexture);
    
    if (sizeX != _texSizeX || sizeY != _texSizeY || sizeZ != _texSizeZ) {
        // Recreate texture with new size
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, sizeX, sizeY, sizeZ, 0, 
                     GL_RED, GL_FLOAT, grid.data());
        
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
        
        _texSizeX = sizeX;
        _texSizeY = sizeY;
        _texSizeZ = sizeZ;
    } else {
        // Just update the data
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, sizeX, sizeY, sizeZ,
                        GL_RED, GL_FLOAT, grid.data());
    }
    
    glBindTexture(GL_TEXTURE_3D, 0);
}

void VolumeRenderer::render(const glm::mat4& view, const glm::mat4& projection,
                            const glm::vec3& cameraPos, float density) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);  // Render both sides of cube
    
    glUseProgram(_shaderProgram);
    
    glUniformMatrix4fv(_locView, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(_locProjection, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3fv(_locCameraPos, 1, glm::value_ptr(cameraPos));
    glUniform1f(_locDensity, density);
    glUniform3f(_locVolumeSize, (float)_texSizeX, (float)_texSizeY, (float)_texSizeZ);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, _volumeTexture);
    glUniform1i(_locVolumeTex, 0);
    
    glBindVertexArray(_vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    
    glBindVertexArray(0);
    glUseProgram(0);
    glEnable(GL_CULL_FACE);
}
