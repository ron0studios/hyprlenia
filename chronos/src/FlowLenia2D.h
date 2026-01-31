#pragma once

#include <glad/gl.h>
#include <string>
#include <vector>

class FlowLenia2D {
public:
    FlowLenia2D();
    ~FlowLenia2D();
    
    void init(int width, int height);
    void shutdown();
    void update();
    void reset();
    void addBlob(float x, float y, float radius);
    void addObstacle(float x, float y, float radius);
    void clearObstacles();
    
    GLuint getTexture() const { return _stateTextures[_currentTex]; }
    GLuint getFlowTexture() const { return _flowTexture; }
    int getWidth() const { return _width; }
    int getHeight() const { return _height; }
    
    // Parameters from Flow Lenia paper (Table 1)
    float R = 15.0f;           // Neighborhood radius [2, 25]
    float T = 5.0f;            // Time resolution (dt = 1/T = 0.2)
    float baseNoise = 0.5f;    // Initial noise level
    
private:
    bool loadShader(const std::string& path, GLuint& program);
    std::string loadShaderSource(const std::string& path);
    
    int _width = 0;
    int _height = 0;
    
    // Compute shader
    GLuint _flowLeniaProgram = 0;
    
    // Textures - double buffered state + flow
    GLuint _stateTextures[2] = {0, 0};
    GLuint _flowTexture = 0;
    int _currentTex = 0;
    
    // Uniform locations
    GLint _locR = -1;
    GLint _locDt = -1;
    GLint _locFlowStrength = -1;
    GLint _locPass = -1;
};
