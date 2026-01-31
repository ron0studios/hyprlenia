#pragma once

#include <glad/gl.h>
#include <string>

class BloomEffect {
public:
    BloomEffect();
    ~BloomEffect();
    
    void init(int width, int height);
    void shutdown();
    void resize(int width, int height);
    
    // Process: extract bright areas, blur, and output
    void process(GLuint inputTexture);
    
    GLuint getBloomTexture() const { return _blurTextures[0]; }
    
    float threshold = 0.3f;
    int blurPasses = 4;
    
private:
    bool loadShader(const std::string& path, GLuint& program);
    
    int _width = 0;
    int _height = 0;
    
    GLuint _extractProgram = 0;
    GLuint _blurProgram = 0;
    
    // Textures for bloom processing
    GLuint _brightTexture = 0;
    GLuint _blurTextures[2] = {0, 0};
    
    // Uniform locations
    GLint _extractThreshold = -1;
    GLint _blurHorizontal = -1;
};
