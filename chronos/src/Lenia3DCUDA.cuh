#pragma once

#include <cuda_runtime.h>
#include <vector>

// Lenia parameters for 3D - based on the "species" from the shader
struct LeniaSpecies {
    float R = 6.0f;         // Kernel radius (space resolution) - smaller for 3D performance
    float T = 8.0f;         // Time resolution (divisions per unit time)
    float baseNoise = 0.3f; // Initial noise level
    
    // Multi-ring kernel parameters (up to 3 rings)
    int betaLen[3] = {2, 3, 1};     // Number of rings per kernel
    float beta[3][3] = {            // Ring heights [kernel][ring]
        {0.25f, 1.0f, 0.0f},
        {1.0f, 0.75f, 0.75f},
        {1.0f, 0.0f, 0.0f}
    };
    
    // Growth function parameters
    float mu[3] = {0.16f, 0.22f, 0.28f};       // Growth centers
    float sigma[3] = {0.105f, 0.042f, 0.025f}; // Growth widths
    float eta[3] = {2.0f, 2.0f, 2.0f};         // Growth strengths
};

class Lenia3DCUDA {
public:
    Lenia3DCUDA();
    ~Lenia3DCUDA();
    
    // Initialize with grid dimensions
    void init(int sizeX, int sizeY, int sizeZ);
    void shutdown();
    
    // Simulation
    void update();
    void reset();
    void addBlob(float x, float y, float z, float radius);
    
    // Get grid data for rendering
    const std::vector<float>& getGrid();
    
    // Parameters
    LeniaSpecies species;
    
    int getSizeX() const { return _sizeX; }
    int getSizeY() const { return _sizeY; }
    int getSizeZ() const { return _sizeZ; }
    
private:
    void copyParamsToDevice();
    
    // Grid dimensions
    int _sizeX = 0, _sizeY = 0, _sizeZ = 0;
    
    // Device memory - double buffered
    float* _dGrid = nullptr;
    float* _dGridNext = nullptr;
    
    // Host memory
    std::vector<float> _hostGrid;
    
    bool _needsCopy = true;
};
