#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>

constexpr int CUDA_NUM_COLORS = 6;

struct ParticleCUDA {
    float3 position;
    float3 velocity;
    int colorIndex;
};

class ParticleLifeCUDA {
public:
    ParticleLifeCUDA();
    ~ParticleLifeCUDA();

    void init(int particleCount);
    void shutdown();
    void update(float deltaTime);
    void randomizeRules();
    
    // Get particles for rendering (copies from GPU)
    const std::vector<ParticleCUDA>& getParticles();
    
    // Simulation parameters (accessible from host)
    float friction = 0.1f;
    float forceFactor = 5.0f;
    float interactionRadius = 0.5f;
    float worldSize = 3.0f;
    
    // Get/set rules
    float getRule(int i, int j) const { return _hostRules[i * CUDA_NUM_COLORS + j]; }
    void setRule(int i, int j, float value);
    
    int getParticleCount() const { return _particleCount; }

private:
    void copyRulesToDevice();
    
    // Device memory
    ParticleCUDA* _dParticles = nullptr;
    float* _dRules = nullptr;
    
    // Host memory
    std::vector<ParticleCUDA> _hostParticles;
    float _hostRules[CUDA_NUM_COLORS * CUDA_NUM_COLORS];
    
    int _particleCount = 0;
    bool _rulesNeedUpdate = true;
};

// Color definitions for rendering
inline glm::vec3 getParticleColor(int colorIndex) {
    static const glm::vec3 colors[CUDA_NUM_COLORS] = {
        {1.0f, 0.2f, 0.3f},   // Red
        {0.2f, 1.0f, 0.4f},   // Green
        {0.3f, 0.5f, 1.0f},   // Blue
        {1.0f, 1.0f, 0.2f},   // Yellow
        {1.0f, 0.4f, 1.0f},   // Magenta
        {0.3f, 1.0f, 1.0f}    // Cyan
    };
    return colors[colorIndex % CUDA_NUM_COLORS];
}
