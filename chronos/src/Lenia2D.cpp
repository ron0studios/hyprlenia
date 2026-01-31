#include "Lenia2D.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <cmath>

Lenia2D::Lenia2D() = default;

Lenia2D::~Lenia2D() {
    shutdown();
}

std::string Lenia2D::loadShaderSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader: " << path << std::endl;
        return "";
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

bool Lenia2D::loadComputeShader() {
    std::string source = loadShaderSource("shaders/lenia_compute.glsl");
    if (source.empty()) {
        // Try relative to executable
        source = loadShaderSource("../shaders/lenia_compute.glsl");
    }
    if (source.empty()) {
        std::cerr << "Could not load compute shader" << std::endl;
        return false;
    }
    
    const char* src = source.c_str();
    
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Compute shader compile error: " << log << std::endl;
        glDeleteShader(shader);
        return false;
    }
    
    _computeProgram = glCreateProgram();
    glAttachShader(_computeProgram, shader);
    glLinkProgram(_computeProgram);
    
    glGetProgramiv(_computeProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(_computeProgram, 512, nullptr, log);
        std::cerr << "Compute program link error: " << log << std::endl;
        glDeleteProgram(_computeProgram);
        glDeleteShader(shader);
        return false;
    }
    
    glDeleteShader(shader);
    
    _locR = glGetUniformLocation(_computeProgram, "R");
    _locDt = glGetUniformLocation(_computeProgram, "dt");
    
    return true;
}

void Lenia2D::init(int width, int height) {
    shutdown();
    
    _width = width;
    _height = height;
    
    // Load compute shader
    if (!loadComputeShader()) {
        std::cerr << "Failed to load compute shader!" << std::endl;
        return;
    }
    
    // Create textures
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &_textures[i]);
        glBindTexture(GL_TEXTURE_2D, _textures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    }
    
    reset();
    
    std::cout << "Lenia 2D initialized: " << width << "x" << height << std::endl;
}

void Lenia2D::shutdown() {
    if (_computeProgram) {
        glDeleteProgram(_computeProgram);
        _computeProgram = 0;
    }
    for (int i = 0; i < 2; i++) {
        if (_textures[i]) {
            glDeleteTextures(1, &_textures[i]);
            _textures[i] = 0;
        }
    }
}

void Lenia2D::reset() {
    if (!_textures[0]) return;
    
    // Generate initial state with noise and blobs
    std::vector<float> data(_width * _height * 4);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Fill with noise
    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            int idx = (y * _width + x) * 4;
            float noise = dis(gen) * baseNoise * 0.5f;
            data[idx + 0] = noise;  // R
            data[idx + 1] = noise;  // G
            data[idx + 2] = noise;  // B
            data[idx + 3] = 1.0f;   // A
        }
    }
    
    // Add initial blobs
    auto addBlobData = [&](float cx, float cy, float r) {
        for (int y = 0; y < _height; y++) {
            for (int x = 0; x < _width; x++) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = std::sqrt(dx*dx + dy*dy);
                if (dist < r) {
                    int idx = (y * _width + x) * 4;
                    float t = dist / r;
                    float val = (1.0f - t * t) * 0.9f;
                    data[idx + 0] = std::max(data[idx + 0], val);
                    data[idx + 1] = std::max(data[idx + 1], val);
                    data[idx + 2] = std::max(data[idx + 2], val);
                }
            }
        }
    };
    
    // Add several starting blobs
    float cx = _width / 2.0f;
    float cy = _height / 2.0f;
    float r = _width / 8.0f;
    
    addBlobData(cx, cy, r);
    addBlobData(cx + r * 1.5f, cy, r * 0.7f);
    addBlobData(cx - r * 1.5f, cy + r * 0.5f, r * 0.6f);
    addBlobData(cx, cy - r * 1.2f, r * 0.5f);
    
    // Upload to both textures
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, _textures[i]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
    }
    
    _currentTex = 0;
}

void Lenia2D::addBlob(float x, float y, float radius) {
    // Read current texture, modify, write back
    std::vector<float> data(_width * _height * 4);
    
    glBindTexture(GL_TEXTURE_2D, _textures[_currentTex]);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, data.data());
    
    for (int py = 0; py < _height; py++) {
        for (int px = 0; px < _width; px++) {
            float dx = px - x;
            float dy = py - y;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist < radius) {
                int idx = (py * _width + px) * 4;
                float t = dist / radius;
                float val = (1.0f - t * t) * 0.9f;
                data[idx + 0] = std::max(data[idx + 0], val);
                data[idx + 1] = std::max(data[idx + 1], val);
                data[idx + 2] = std::max(data[idx + 2], val);
            }
        }
    }
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
}

void Lenia2D::update() {
    if (!_computeProgram) return;
    
    glUseProgram(_computeProgram);
    
    // Set uniforms
    glUniform1f(_locR, R);
    glUniform1f(_locDt, 1.0f / T);
    
    // Bind textures
    int nextTex = 1 - _currentTex;
    glBindImageTexture(0, _textures[_currentTex], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, _textures[nextTex], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    
    // Dispatch compute shader
    GLuint groupsX = (_width + 15) / 16;
    GLuint groupsY = (_height + 15) / 16;
    glDispatchCompute(groupsX, groupsY, 1);
    
    // Memory barrier
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    
    // Swap buffers
    _currentTex = nextTex;
}
