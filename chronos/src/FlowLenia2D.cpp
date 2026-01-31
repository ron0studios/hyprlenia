#include "FlowLenia2D.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <cmath>

FlowLenia2D::FlowLenia2D() = default;

FlowLenia2D::~FlowLenia2D() {
    shutdown();
}

std::string FlowLenia2D::loadShaderSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        // Try alternative paths
        std::vector<std::string> paths = {
            path,
            "../" + path,
            "shaders/" + path,
            "../shaders/" + path
        };
        for (const auto& p : paths) {
            file.open(p);
            if (file.is_open()) break;
        }
    }
    if (!file.is_open()) {
        std::cerr << "Failed to open shader: " << path << std::endl;
        return "";
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

bool FlowLenia2D::loadShader(const std::string& path, GLuint& program) {
    std::string source = loadShaderSource(path);
    if (source.empty()) return false;
    
    const char* src = source.c_str();
    
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Compute shader compile error (" << path << "): " << log << std::endl;
        glDeleteShader(shader);
        return false;
    }
    
    program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(program, 1024, nullptr, log);
        std::cerr << "Compute program link error: " << log << std::endl;
        glDeleteProgram(program);
        glDeleteShader(shader);
        return false;
    }
    
    glDeleteShader(shader);
    return true;
}

void FlowLenia2D::init(int width, int height) {
    shutdown();
    
    _width = width;
    _height = height;
    
    // Load compute shader
    if (!loadShader("shaders/flow_lenia.glsl", _flowLeniaProgram)) {
        std::cerr << "Failed to load flow_lenia.glsl!" << std::endl;
        return;
    }
    
    // Get uniform locations
    _locR = glGetUniformLocation(_flowLeniaProgram, "R");
    _locDt = glGetUniformLocation(_flowLeniaProgram, "dt");
    _locFlowStrength = glGetUniformLocation(_flowLeniaProgram, "flowStrength");
    _locPass = glGetUniformLocation(_flowLeniaProgram, "pass");
    
    // Create state textures (double buffered)
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &_stateTextures[i]);
        glBindTexture(GL_TEXTURE_2D, _stateTextures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    }
    
    // Create flow texture
    glGenTextures(1, &_flowTexture);
    glBindTexture(GL_TEXTURE_2D, _flowTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, nullptr);
    
    reset();
    
    std::cout << "Flow Lenia 2D initialized: " << width << "x" << height << std::endl;
}

void FlowLenia2D::shutdown() {
    if (_flowLeniaProgram) {
        glDeleteProgram(_flowLeniaProgram);
        _flowLeniaProgram = 0;
    }
    for (int i = 0; i < 2; i++) {
        if (_stateTextures[i]) {
            glDeleteTextures(1, &_stateTextures[i]);
            _stateTextures[i] = 0;
        }
    }
    if (_flowTexture) {
        glDeleteTextures(1, &_flowTexture);
        _flowTexture = 0;
    }
}

void FlowLenia2D::reset() {
    if (!_stateTextures[0]) return;
    
    std::vector<float> data(_width * _height * 4, 0.0f);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Rule 388 initialization: random 50% noise (binary)
    // This creates the initial chaos that self-organizes
    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            int idx = (y * _width + x) * 4;
            float val = dis(gen) > 0.5f ? 1.0f : 0.0f;
            data[idx + 0] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
            data[idx + 3] = 1.0f;
        }
    }
    
    // Upload to both textures
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, _stateTextures[i]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
    }
    
    _currentTex = 0;
}

void FlowLenia2D::addBlob(float x, float y, float radius) {
    std::vector<float> data(_width * _height * 4);
    
    glBindTexture(GL_TEXTURE_2D, _stateTextures[_currentTex]);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, data.data());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Single-channel Lenia blob with smooth Gaussian profile
    float actualRadius = radius * 0.3f;
    
    for (int py = 0; py < _height; py++) {
        for (int px = 0; px < _width; px++) {
            float dx = px - x;
            float dy = py - y;
            float dist = std::sqrt(dx*dx + dy*dy);
            
            int idx = (py * _width + px) * 4;
            
            if (dist < actualRadius) {
                float r = dist / actualRadius;
                // Gaussian-like profile
                float intensity = std::exp(-4.0f * r * r);
                data[idx + 0] = std::max(data[idx + 0], intensity * (0.5f + dis(gen) * 0.5f));
            }
        }
    }
    
    // Update both textures
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
    glBindTexture(GL_TEXTURE_2D, _stateTextures[1 - _currentTex]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
}

void FlowLenia2D::addObstacle(float x, float y, float radius) {
    std::vector<float> data(_width * _height * 4);
    
    glBindTexture(GL_TEXTURE_2D, _stateTextures[_currentTex]);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, data.data());
    
    for (int py = 0; py < _height; py++) {
        for (int px = 0; px < _width; px++) {
            float dx = px - x;
            float dy = py - y;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist < radius) {
                int idx = (py * _width + px) * 4;
                data[idx + 0] = 0.0f;  // Clear density
                data[idx + 3] = 1.0f;  // Set obstacle
            }
        }
    }
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
    
    // Also update other buffer
    int other = 1 - _currentTex;
    glBindTexture(GL_TEXTURE_2D, _stateTextures[other]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
}

void FlowLenia2D::clearObstacles() {
    for (int i = 0; i < 2; i++) {
        std::vector<float> data(_width * _height * 4);
        glBindTexture(GL_TEXTURE_2D, _stateTextures[i]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, data.data());
        
        for (int j = 3; j < _width * _height * 4; j += 4) {
            data[j] = 0.0f;  // Clear obstacle flag
        }
        
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, data.data());
    }
}

void FlowLenia2D::update() {
    if (!_flowLeniaProgram) return;
    
    glUseProgram(_flowLeniaProgram);
    
    float dt = 1.0f / T;
    glUniform1f(_locR, R);
    glUniform1f(_locDt, dt);
    
    GLuint groupsX = (_width + 15) / 16;
    GLuint groupsY = (_height + 15) / 16;
    
    int nextTex = 1 - _currentTex;
    
    // Pass 0: Compute growth and flow field
    glUniform1i(_locPass, 0);
    glBindImageTexture(0, _stateTextures[_currentTex], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, _stateTextures[nextTex], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(2, _flowTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    
    // Swap for pass 1
    _currentTex = nextTex;
    nextTex = 1 - _currentTex;
    
    // Pass 1: Apply growth and advection
    glUniform1i(_locPass, 1);
    glBindImageTexture(0, _stateTextures[_currentTex], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, _stateTextures[nextTex], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(2, _flowTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    
    _currentTex = nextTex;
}
