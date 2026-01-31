#include "BloomEffect.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

BloomEffect::BloomEffect() = default;
BloomEffect::~BloomEffect() { shutdown(); }

static std::string loadFile(const std::string& path) {
    std::vector<std::string> paths = {
        path, "../" + path, "shaders/" + path, "../shaders/" + path
    };
    for (const auto& p : paths) {
        std::ifstream file(p);
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }
    }
    return "";
}

bool BloomEffect::loadShader(const std::string& path, GLuint& program) {
    std::string source = loadFile(path);
    if (source.empty()) {
        std::cerr << "Failed to load shader: " << path << std::endl;
        return false;
    }
    
    const char* src = source.c_str();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Shader compile error (" << path << "): " << log << std::endl;
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
        std::cerr << "Program link error: " << log << std::endl;
        glDeleteProgram(program);
        program = 0;
        glDeleteShader(shader);
        return false;
    }
    
    glDeleteShader(shader);
    return true;
}

void BloomEffect::init(int width, int height) {
    shutdown();
    
    _width = width;
    _height = height;
    
    loadShader("shaders/bloom_extract.glsl", _extractProgram);
    loadShader("shaders/blur_compute.glsl", _blurProgram);
    
    if (_extractProgram) {
        _extractThreshold = glGetUniformLocation(_extractProgram, "threshold");
    }
    if (_blurProgram) {
        _blurHorizontal = glGetUniformLocation(_blurProgram, "horizontal");
    }
    
    // Create textures at half resolution for performance
    int halfW = width / 2;
    int halfH = height / 2;
    
    glGenTextures(1, &_brightTexture);
    glBindTexture(GL_TEXTURE_2D, _brightTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, halfW, halfH, 0, GL_RGBA, GL_FLOAT, nullptr);
    
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &_blurTextures[i]);
        glBindTexture(GL_TEXTURE_2D, _blurTextures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, halfW, halfH, 0, GL_RGBA, GL_FLOAT, nullptr);
    }
    
    std::cout << "Bloom effect initialized at " << halfW << "x" << halfH << std::endl;
}

void BloomEffect::shutdown() {
    if (_extractProgram) { glDeleteProgram(_extractProgram); _extractProgram = 0; }
    if (_blurProgram) { glDeleteProgram(_blurProgram); _blurProgram = 0; }
    if (_brightTexture) { glDeleteTextures(1, &_brightTexture); _brightTexture = 0; }
    for (int i = 0; i < 2; i++) {
        if (_blurTextures[i]) { glDeleteTextures(1, &_blurTextures[i]); _blurTextures[i] = 0; }
    }
}

void BloomEffect::resize(int width, int height) {
    if (width != _width || height != _height) {
        init(width, height);
    }
}

void BloomEffect::process(GLuint inputTexture) {
    if (!_extractProgram || !_blurProgram) return;
    
    int halfW = _width / 2;
    int halfH = _height / 2;
    GLuint groupsX = (halfW + 15) / 16;
    GLuint groupsY = (halfH + 15) / 16;
    
    // Step 1: Extract bright areas
    glUseProgram(_extractProgram);
    glUniform1f(_extractThreshold, threshold);
    glBindImageTexture(0, inputTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, _brightTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    
    // Copy bright to blur texture 0
    glCopyImageSubData(_brightTexture, GL_TEXTURE_2D, 0, 0, 0, 0,
                       _blurTextures[0], GL_TEXTURE_2D, 0, 0, 0, 0,
                       halfW, halfH, 1);
    
    // Step 2: Blur passes (ping-pong between blur textures)
    glUseProgram(_blurProgram);
    
    for (int pass = 0; pass < blurPasses; pass++) {
        // Horizontal blur
        glUniform1i(_blurHorizontal, 1);
        glBindImageTexture(0, _blurTextures[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, _blurTextures[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        // Vertical blur
        glUniform1i(_blurHorizontal, 0);
        glBindImageTexture(0, _blurTextures[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, _blurTextures[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    }
}
