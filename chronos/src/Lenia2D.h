#pragma once

#include <glad/gl.h>
#include <string>
#include <vector>

class Lenia2D {
public:
    Lenia2D();
    ~Lenia2D();
    
    void init(int width, int height);
    void shutdown();
    void update();
    void reset();
    void addBlob(float x, float y, float radius);
    
    GLuint getTexture() const { return _textures[_currentTex]; }
    int getWidth() const { return _width; }
    int getHeight() const { return _height; }
    
    // Parameters
    float R = 8.0f;       // Kernel radius
    float T = 10.0f;      // Time resolution
    float baseNoise = 0.5f;
    
private:
    bool loadComputeShader();
    std::string loadShaderSource(const std::string& path);
    
    int _width = 0;
    int _height = 0;
    
    GLuint _computeProgram = 0;
    GLuint _textures[2] = {0, 0};  // Double buffer
    int _currentTex = 0;
    
    // Uniform locations
    GLint _locR = -1;
    GLint _locDt = -1;
};
