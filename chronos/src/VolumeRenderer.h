#pragma once

#include <glm/glm.hpp>
#include <vector>

/**
 * VolumeRenderer - Renders 3D Lenia grid as volumetric data using ray marching
 */
class VolumeRenderer {
public:
    VolumeRenderer();
    ~VolumeRenderer();
    
    void init();
    void shutdown();
    
    // Update the 3D texture from grid data
    void updateTexture(const std::vector<float>& grid, int sizeX, int sizeY, int sizeZ);
    
    // Render the volume
    void render(const glm::mat4& view, const glm::mat4& projection, 
                const glm::vec3& cameraPos, float density = 1.0f);
    
private:
    unsigned int _shaderProgram = 0;
    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _volumeTexture = 0;
    
    int _texSizeX = 0, _texSizeY = 0, _texSizeZ = 0;
    
    // Uniform locations
    int _locView = -1;
    int _locProjection = -1;
    int _locCameraPos = -1;
    int _locVolumeTex = -1;
    int _locDensity = -1;
    int _locVolumeSize = -1;
};
