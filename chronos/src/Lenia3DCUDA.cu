#include "Lenia3DCUDA.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// Device constants
__constant__ float d_R;
__constant__ float d_dt;
__constant__ float d_mu[3];
__constant__ float d_sigma[3];
__constant__ float d_eta[3];

// Bell-shaped Gaussian curve
__device__ __forceinline__ float bell(float x, float mu, float sigma) {
    float diff = x - mu;
    return expf(-diff * diff / (2.0f * sigma * sigma));
}

// Optimized Lenia kernel with sparse sampling
__global__ void leniaUpdateKernel(
    float* __restrict__ gridOut, 
    const float* __restrict__ gridIn,
    int sizeX, int sizeY, int sizeZ,
    float R)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= sizeX || y >= sizeY || z >= sizeZ) return;
    
    int idx = z * sizeY * sizeX + y * sizeX + x;
    
    int intR = (int)R;
    float invR = 1.0f / R;
    
    // Use sparse sampling - sample at shell radii instead of every voxel
    float sum = 0.0f;
    float totalWeight = 0.0001f;
    
    // Sample center
    sum += gridIn[idx] * 1.0f;
    totalWeight += 1.0f;
    
    // Sample at discrete shell distances (sparse sampling)
    const int numShells = 4;
    const float shellRadii[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    const float shellWeights[4] = {0.9f, 0.7f, 0.4f, 0.1f};
    
    // 6 face directions + 8 corner directions + 12 edge directions = 26 directions
    const int dirs[26][3] = {
        // 6 faces
        {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
        // 8 corners
        {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}, {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1},
        // 12 edges
        {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0},
        {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1},
        {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1}
    };
    
    #pragma unroll
    for (int s = 0; s < numShells; s++) {
        float r = shellRadii[s] * R;
        float w = shellWeights[s];
        
        #pragma unroll
        for (int d = 0; d < 26; d++) {
            int dx = (int)(dirs[d][0] * r);
            int dy = (int)(dirs[d][1] * r);
            int dz = (int)(dirs[d][2] * r);
            
            // Wrap coordinates
            int nx = (x + dx + sizeX) % sizeX;
            int ny = (y + dy + sizeY) % sizeY;
            int nz = (z + dz + sizeZ) % sizeZ;
            
            int nidx = nz * sizeY * sizeX + ny * sizeX + nx;
            sum += gridIn[nidx] * w;
            totalWeight += w;
        }
    }
    
    float avg = sum / totalWeight;
    
    // Growth function - simplified for speed
    float g1 = bell(avg, d_mu[0], d_sigma[0]);
    float g2 = bell(avg, d_mu[1], d_sigma[1]);
    float growth = d_eta[0] * (2.0f * fmaxf(g1, g2) - 1.0f);
    
    // Update
    float current = gridIn[idx];
    float newVal = current + d_dt * growth;
    gridOut[idx] = fminf(fmaxf(newVal, 0.0f), 1.0f);
}

// Fast initialization
__global__ void initNoiseKernel(
    float* grid,
    int sizeX, int sizeY, int sizeZ,
    float baseNoise,
    unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= sizeX || y >= sizeY || z >= sizeZ) return;
    
    int idx = z * sizeY * sizeX + y * sizeX + x;
    
    unsigned int h = seed + x * 374761393u + y * 668265263u + z * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    
    float noise = (float)(h % 10000) / 10000.0f;
    grid[idx] = baseNoise * noise * 0.5f;
}

// Add blob
__global__ void addBlobKernel(
    float* grid,
    int sizeX, int sizeY, int sizeZ,
    float blobX, float blobY, float blobZ, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= sizeX || y >= sizeY || z >= sizeZ) return;
    
    int idx = z * sizeY * sizeX + y * sizeX + x;
    
    float dx = x - blobX;
    float dy = y - blobY;
    float dz = z - blobZ;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    
    if (dist < radius) {
        float t = dist / radius;
        float value = (1.0f - t * t) * 0.9f;
        grid[idx] = fmaxf(grid[idx], value);
    }
}

// Host implementation
Lenia3DCUDA::Lenia3DCUDA() = default;

Lenia3DCUDA::~Lenia3DCUDA() {
    shutdown();
}

void Lenia3DCUDA::init(int sizeX, int sizeY, int sizeZ) {
    shutdown();
    
    _sizeX = sizeX;
    _sizeY = sizeY;
    _sizeZ = sizeZ;
    
    size_t gridSize = sizeX * sizeY * sizeZ * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&_dGrid, gridSize));
    CUDA_CHECK(cudaMalloc(&_dGridNext, gridSize));
    
    _hostGrid.resize(sizeX * sizeY * sizeZ);
    
    copyParamsToDevice();
    reset();
    
    printf("Lenia 3D CUDA initialized: %dx%dx%d grid (optimized)\n", sizeX, sizeY, sizeZ);
}

void Lenia3DCUDA::shutdown() {
    if (_dGrid) {
        cudaFree(_dGrid);
        _dGrid = nullptr;
    }
    if (_dGridNext) {
        cudaFree(_dGridNext);
        _dGridNext = nullptr;
    }
    _hostGrid.clear();
}

void Lenia3DCUDA::copyParamsToDevice() {
    float dt = 1.0f / species.T;
    CUDA_CHECK(cudaMemcpyToSymbol(d_R, &species.R, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dt, &dt, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mu, species.mu, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_sigma, species.sigma, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_eta, species.eta, 3 * sizeof(float)));
}

void Lenia3DCUDA::reset() {
    if (!_dGrid) return;
    
    dim3 blockSize(8, 8, 8);
    dim3 numBlocks(
        (_sizeX + blockSize.x - 1) / blockSize.x,
        (_sizeY + blockSize.y - 1) / blockSize.y,
        (_sizeZ + blockSize.z - 1) / blockSize.z
    );
    
    unsigned int seed = (unsigned int)time(nullptr);
    initNoiseKernel<<<numBlocks, blockSize>>>(_dGrid, _sizeX, _sizeY, _sizeZ, species.baseNoise, seed);
    CUDA_CHECK(cudaGetLastError());
    
    // Add initial structure
    float cx = _sizeX / 2.0f;
    float cy = _sizeY / 2.0f;
    float cz = _sizeZ / 2.0f;
    float r = _sizeX / 5.0f;
    
    addBlobKernel<<<numBlocks, blockSize>>>(_dGrid, _sizeX, _sizeY, _sizeZ, cx, cy, cz, r);
    addBlobKernel<<<numBlocks, blockSize>>>(_dGrid, _sizeX, _sizeY, _sizeZ, cx + r*0.8f, cy, cz, r * 0.6f);
    addBlobKernel<<<numBlocks, blockSize>>>(_dGrid, _sizeX, _sizeY, _sizeZ, cx - r*0.8f, cy + r*0.5f, cz, r * 0.5f);
    CUDA_CHECK(cudaGetLastError());
    
    _needsCopy = true;
}

void Lenia3DCUDA::addBlob(float x, float y, float z, float radius) {
    if (!_dGrid) return;
    
    dim3 blockSize(8, 8, 8);
    dim3 numBlocks(
        (_sizeX + blockSize.x - 1) / blockSize.x,
        (_sizeY + blockSize.y - 1) / blockSize.y,
        (_sizeZ + blockSize.z - 1) / blockSize.z
    );
    
    addBlobKernel<<<numBlocks, blockSize>>>(_dGrid, _sizeX, _sizeY, _sizeZ, x, y, z, radius);
    CUDA_CHECK(cudaGetLastError());
    
    _needsCopy = true;
}

void Lenia3DCUDA::update() {
    if (!_dGrid) return;
    
    copyParamsToDevice();
    
    dim3 blockSize(8, 8, 8);
    dim3 numBlocks(
        (_sizeX + blockSize.x - 1) / blockSize.x,
        (_sizeY + blockSize.y - 1) / blockSize.y,
        (_sizeZ + blockSize.z - 1) / blockSize.z
    );
    
    leniaUpdateKernel<<<numBlocks, blockSize>>>(_dGridNext, _dGrid, _sizeX, _sizeY, _sizeZ, species.R);
    CUDA_CHECK(cudaGetLastError());
    
    // Swap buffers
    float* tmp = _dGrid;
    _dGrid = _dGridNext;
    _dGridNext = tmp;
    
    _needsCopy = true;
}

const std::vector<float>& Lenia3DCUDA::getGrid() {
    if (_needsCopy && _dGrid) {
        size_t gridSize = _sizeX * _sizeY * _sizeZ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(_hostGrid.data(), _dGrid, gridSize, cudaMemcpyDeviceToHost));
        _needsCopy = false;
    }
    return _hostGrid;
}
