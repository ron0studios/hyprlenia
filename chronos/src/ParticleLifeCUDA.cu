#include "ParticleLifeCUDA.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <random>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// Device constants
__constant__ float d_rules[CUDA_NUM_COLORS * CUDA_NUM_COLORS];
__constant__ float d_friction;
__constant__ float d_forceFactor;
__constant__ float d_interactionRadius;
__constant__ float d_worldSize;

// Force calculation kernel - each thread handles one particle
__global__ void updateParticlesKernel(ParticleCUDA* particles, int particleCount, float deltaTime) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleCount) return;
    
    ParticleCUDA& p = particles[i];
    float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);
    
    float radiusSq = d_interactionRadius * d_interactionRadius;
    
    // Calculate forces from all other particles
    for (int j = 0; j < particleCount; ++j) {
        if (i == j) continue;
        
        const ParticleCUDA& other = particles[j];
        
        float3 diff = make_float3(
            other.position.x - p.position.x,
            other.position.y - p.position.y,
            other.position.z - p.position.z
        );
        
        // Wrap around periodic boundaries
        float halfWorld = d_worldSize * 0.5f;
        if (diff.x > halfWorld) diff.x -= d_worldSize;
        if (diff.x < -halfWorld) diff.x += d_worldSize;
        if (diff.y > halfWorld) diff.y -= d_worldSize;
        if (diff.y < -halfWorld) diff.y += d_worldSize;
        if (diff.z > halfWorld) diff.z -= d_worldSize;
        if (diff.z < -halfWorld) diff.z += d_worldSize;
        
        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
        if (distSq > radiusSq || distSq < 0.0001f) continue;
        
        float dist = sqrtf(distSq);
        float normalizedDist = dist / d_interactionRadius;
        
        // Get attraction rule
        float attraction = d_rules[p.colorIndex * CUDA_NUM_COLORS + other.colorIndex];
        
        // Particle life force curve
        float force = 0.0f;
        const float beta = 0.3f;
        
        if (normalizedDist < beta) {
            // Universal repulsion zone
            force = normalizedDist / beta - 1.0f;
        } else {
            // Attraction/repulsion based on rule
            force = attraction * (1.0f - fabsf(2.0f * normalizedDist - 1.0f - beta) / (1.0f - beta));
        }
        
        // Normalize direction and apply force
        float invDist = 1.0f / dist;
        totalForce.x += diff.x * invDist * force;
        totalForce.y += diff.y * invDist * force;
        totalForce.z += diff.z * invDist * force;
    }
    
    // Apply force to velocity
    p.velocity.x += totalForce.x * d_forceFactor * deltaTime;
    p.velocity.y += totalForce.y * d_forceFactor * deltaTime;
    p.velocity.z += totalForce.z * d_forceFactor * deltaTime;
    
    // Apply friction
    float frictionMult = 1.0f - d_friction * deltaTime * 10.0f;
    p.velocity.x *= frictionMult;
    p.velocity.y *= frictionMult;
    p.velocity.z *= frictionMult;
    
    // Update position
    p.position.x += p.velocity.x * deltaTime;
    p.position.y += p.velocity.y * deltaTime;
    p.position.z += p.velocity.z * deltaTime;
    
    // Wrap position
    float halfWorld = d_worldSize * 0.5f;
    if (p.position.x > halfWorld) p.position.x -= d_worldSize;
    if (p.position.x < -halfWorld) p.position.x += d_worldSize;
    if (p.position.y > halfWorld) p.position.y -= d_worldSize;
    if (p.position.y < -halfWorld) p.position.y += d_worldSize;
    if (p.position.z > halfWorld) p.position.z -= d_worldSize;
    if (p.position.z < -halfWorld) p.position.z += d_worldSize;
}

// Host class implementation
ParticleLifeCUDA::ParticleLifeCUDA() {
    randomizeRules();
}

ParticleLifeCUDA::~ParticleLifeCUDA() {
    shutdown();
}

void ParticleLifeCUDA::init(int particleCount) {
    shutdown();
    
    _particleCount = particleCount;
    _hostParticles.resize(particleCount);
    
    // Initialize particles on host
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> posDist(-worldSize * 0.5f, worldSize * 0.5f);
    std::uniform_int_distribution<int> colorDist(0, CUDA_NUM_COLORS - 1);
    
    for (int i = 0; i < particleCount; ++i) {
        _hostParticles[i].position = make_float3(posDist(rng), posDist(rng), posDist(rng));
        _hostParticles[i].velocity = make_float3(0.0f, 0.0f, 0.0f);
        _hostParticles[i].colorIndex = colorDist(rng);
    }
    
    // Allocate and copy to device
    CUDA_CHECK(cudaMalloc(&_dParticles, particleCount * sizeof(ParticleCUDA)));
    CUDA_CHECK(cudaMemcpy(_dParticles, _hostParticles.data(), 
                          particleCount * sizeof(ParticleCUDA), cudaMemcpyHostToDevice));
    
    // Allocate rules on device
    CUDA_CHECK(cudaMalloc(&_dRules, CUDA_NUM_COLORS * CUDA_NUM_COLORS * sizeof(float)));
    
    _rulesNeedUpdate = true;
    copyRulesToDevice();
    
    printf("CUDA Particle Life initialized with %d particles\n", particleCount);
}

void ParticleLifeCUDA::shutdown() {
    if (_dParticles) {
        cudaFree(_dParticles);
        _dParticles = nullptr;
    }
    if (_dRules) {
        cudaFree(_dRules);
        _dRules = nullptr;
    }
    _hostParticles.clear();
    _particleCount = 0;
}

void ParticleLifeCUDA::randomizeRules() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> ruleDist(-1.0f, 1.0f);
    
    for (int i = 0; i < CUDA_NUM_COLORS * CUDA_NUM_COLORS; ++i) {
        _hostRules[i] = ruleDist(rng);
    }
    _rulesNeedUpdate = true;
}

void ParticleLifeCUDA::setRule(int i, int j, float value) {
    _hostRules[i * CUDA_NUM_COLORS + j] = value;
    _rulesNeedUpdate = true;
}

void ParticleLifeCUDA::copyRulesToDevice() {
    if (!_rulesNeedUpdate) return;
    
    // Copy to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_rules, _hostRules, 
                                   CUDA_NUM_COLORS * CUDA_NUM_COLORS * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_friction, &friction, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_forceFactor, &forceFactor, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_interactionRadius, &interactionRadius, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_worldSize, &worldSize, sizeof(float)));
    
    _rulesNeedUpdate = false;
}

void ParticleLifeCUDA::update(float deltaTime) {
    if (_particleCount == 0) return;
    
    // Clamp delta time
    deltaTime = fminf(deltaTime, 0.033f);
    
    // Update constants if needed
    _rulesNeedUpdate = true;  // Always update params for now
    copyRulesToDevice();
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (_particleCount + blockSize - 1) / blockSize;
    
    updateParticlesKernel<<<numBlocks, blockSize>>>(_dParticles, _particleCount, deltaTime);
    
    CUDA_CHECK(cudaGetLastError());
}

const std::vector<ParticleCUDA>& ParticleLifeCUDA::getParticles() {
    if (_particleCount > 0) {
        CUDA_CHECK(cudaMemcpy(_hostParticles.data(), _dParticles,
                              _particleCount * sizeof(ParticleCUDA), cudaMemcpyDeviceToHost));
    }
    return _hostParticles;
}
