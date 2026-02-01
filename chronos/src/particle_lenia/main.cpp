/*
 * CHRONOS - Particle Lenia with Evolution
 *
 * An advanced cellular automata simulation featuring:
 * - Particle-based Lenia (continuous game of life)
 * - Multiple species with different parameters
 * - Evolution: particles can reproduce, mutate, and die
 * - Survival mechanics: energy, predation, competition
 *
 * Based on:
 * https://google-research.github.io/self-organising-systems/particle-lenia/
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <sstream>

#include "core/Buffer.h"
#include "core/ComputeShader.h"
#include "core/RenderShader.h"

// Window dimensions
int WINDOW_WIDTH = 1200;
int WINDOW_HEIGHT = 900;

// Simulation parameters
struct SimulationParams {
  // World dimensions (3D cube)
  float worldWidth = 40.0f;
  float worldHeight = 40.0f;
  float worldDepth = 40.0f;

  // Particle count
  int numParticles = 500;
  int maxParticles = 2000;

  // Kernel parameters - controls sensing/interaction range
  float w_k = 0.022f;     // Kernel weight
  float mu_k = 4.0f;      // Kernel peak distance
  float sigma_k2 = 1.0f;  // Kernel width squared

  // Growth parameters - the "Lenia magic"
  float mu_g = 0.6f;         // Optimal density (growth center)
  float sigma_g2 = 0.0225f;  // Growth width squared

  // Repulsion parameters
  float c_rep = 1.0f;  // Repulsion strength

  // Time integration
  float dt = 0.1f;  // Time step
  float h = 0.01f;  // Gradient calculation distance

  // Evolution parameters (disabled by default for stability)
  bool evolutionEnabled = false;
  float birthRate = 0.001f;        // Chance to reproduce per step
  float deathRate = 0.0f;          // Base death rate (0 = no random death)
  float mutationRate = 0.1f;       // Mutation strength
  float energyDecay = 0.0f;        // Energy loss per step (0 = no decay)
  float energyFromGrowth = 0.01f;  // Energy gained from good growth

  // View parameters
  float translateX = 0.0f;
  float translateY = 0.0f;
  float translateZ = 0.0f;
  float zoom = 1.0f;

  // Rendering
  int stepsPerFrame = 5;
  bool showFields = true;
  int fieldType = 3;  // 0=none, 1=U, 2=R, 3=G, 4=E

  // Food system parameters
  bool foodEnabled = true;
  float foodSpawnRate =
      0.002f;  // Probability of food spawning per cell per step
  float foodDecayRate = 0.001f;        // How fast food decays naturally
  float foodMaxAmount = 1.0f;          // Maximum food per cell
  float foodConsumptionRadius = 2.0f;  // How far particles can reach to eat
  bool showFood = true;                // Show food on display

  // 3D Rendering
  bool view3D = true;            // Default to 3D mode
  float cameraAngle = 45.0f;     // Degrees from horizontal
  float cameraRotation = 0.0f;   // Rotation around Y axis
  float cameraDistance = 60.0f;  // Distance from center
  float heightScale = 10.0f;     // Height multiplier for terrain
  float glowIntensity = 1.5f;    // Glow effect strength
  bool showWireframe = false;    // Show terrain wireframe
  float ambientLight = 0.5f;     // Ambient lighting (higher for visibility)
  float particleSize = 20.0f;    // 3D particle size

  // Interaction
  int interactionMode =
      0;  // 0=None, 1=Spawn, 2=Repel, 3=Attract, 4=Spawn Orbium
  float brushRadius = 5.0f;
  float forceStrength = 0.5f;

  // Goal System
  int goalMode = 0;  // 0=None, 1=Circle, 2=Box, 3=Text, 4=Image
  float goalStrength = 0.1f;
  char goalImagePath[256] = "goal.bmp";

  // Render settings
  bool showGoal = false;
};

// Particle structure (must match shader) - 14 floats total
struct Particle {
  float x, y, z;     // Position (3D)
  float vx, vy, vz;  // Velocity (3D)
  float energy;      // Health/energy [0, 1]
  float species;     // Species ID (affects color)
  float age;         // Age in simulation steps
  float dna[5];  // Genetic parameters (mu_k, sigma_k2, mu_g, sigma_g2, c_rep
                 // variations)
};

constexpr int PARTICLE_FLOATS = 14;  // Number of floats per particle

class ParticleLeniaSimulation {
 public:
  SimulationParams params;

  Buffer particleBufferA;
  Buffer particleBufferB;
  bool useBufferA = true;

  ComputeShader stepShader;
  RenderShader displayShader;

  // 3D Rendering resources
  ComputeShader heightmapShader;
  RenderShader terrainShader;
  RenderShader particle3DShader;
  GLuint heightmapTexture = 0;
  GLuint terrainVAO = 0;
  GLuint terrainVBO = 0;
  GLuint terrainEBO = 0;
  GLuint particleVAO = 0;  // Empty VAO for particle rendering with gl_VertexID
  int terrainGridSize = 128;  // 128x128 grid for terrain
  int terrainIndexCount = 0;

  // Food system resources
  ComputeShader foodUpdateShader;
  GLuint foodTexture = 0;
  int foodGridSize = 128;  // 128x128 food grid

  // Goal system resources
  GLuint goalTexture = 0;
  int goalGridSize = 512;

  std::mt19937 rng;

  // Stats
  int aliveCount = 0;
  float avgEnergy = 0.0f;
  float avgAge = 0.0f;

  void init() {
    // Initialize RNG
    rng = std::mt19937(std::random_device{}());

    // Calculate buffer size: each particle has 14 floats (3D)
    int bufferSize = params.maxParticles * PARTICLE_FLOATS;

    particleBufferA = Buffer(bufferSize, GL_SHADER_STORAGE_BUFFER);
    particleBufferB = Buffer(bufferSize, GL_SHADER_STORAGE_BUFFER);

    particleBufferA.init();
    particleBufferB.init();

    // Initialize particles
    resetParticles();

    // Load shaders
    stepShader = ComputeShader("shaders/particle_lenia_step.comp");
    stepShader.init();

    displayShader = RenderShader("shaders/passthrough.vert",
                                 "shaders/particle_lenia_display.frag");
    displayShader.init();

    // Initialize 3D rendering
    init3D();

    // Initialize food system
    initFood();

    // Initialize goal system
    initGoal();
  }

  void initGoal() {
    glGenTextures(1, &goalTexture);
    glBindTexture(GL_TEXTURE_2D, goalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, goalGridSize, goalGridSize, 0,
                 GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    updateGoalTexture();
  }

  bool loadBMP(const char* filename, std::vector<float>& outData, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      std::cout << "Failed to open image: " << filename << std::endl;
      return false;
    }
    unsigned char header[54];
    if (!file.read(reinterpret_cast<char*>(header), 54)) return false;
    if (header[0] != 'B' || header[1] != 'M') return false;
    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int imageSize = *(int*)&header[34];
    if (imageSize == 0) imageSize = width * height * 3;
    int dataPos = *(int*)&header[10];
    if (dataPos == 0) dataPos = 54;
    std::vector<unsigned char> img(imageSize);
    file.seekg(dataPos);
    file.read(reinterpret_cast<char*>(img.data()), imageSize);
    file.close();

    outData.resize(size * size);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        int srcX = x * width / size;
        int srcY = (size - 1 - y) * height / size;  // Flip Y for GL
        if (srcX >= width) srcX = width - 1;
        if (srcY >= height) srcY = height - 1;
        int idx = (srcY * width + srcX) * 3;
        if (idx < imageSize - 2) {
          float val =
              (img[idx + 2] + img[idx + 1] + img[idx]) / (3.0f * 255.0f);
          outData[y * size + x] = val;
        }
      }
    }
    return true;
  }

  void updateGoalTexture() {
    std::vector<float> data(goalGridSize * goalGridSize, 0.0f);

    if (params.goalMode == 1) {  // Circle
      float cx = goalGridSize / 2.0f;
      float cy = goalGridSize / 2.0f;
      float r = goalGridSize * 0.3f;
      float thickness = goalGridSize * 0.05f;

#pragma omp parallel for collapse(2)
      for (int y = 0; y < goalGridSize; y++) {
        for (int x = 0; x < goalGridSize; x++) {
          float dx = x - cx;
          float dy = y - cy;
          float dist = sqrt(dx * dx + dy * dy);
          float val = exp(-pow(dist - r, 2) / (2.0f * thickness * thickness));
          data[y * goalGridSize + x] = val;
        }
      }
    } else if (params.goalMode == 2) {  // Box
      float margin = goalGridSize * 0.2f;
#pragma omp parallel for collapse(2)
      for (int y = 0; y < goalGridSize; y++) {
        for (int x = 0; x < goalGridSize; x++) {
          if (x > margin && x < goalGridSize - margin && y > margin &&
              y < goalGridSize - margin) {
            float dx =
                std::min(std::min(x - margin, goalGridSize - margin - x),
                         std::min(y - margin, goalGridSize - margin - y));

            if (dx < 20.0f) data[y * goalGridSize + x] = 1.0f;
          }
        }
      }
    } else if (params.goalMode == 3) {  // Text "HI"
      // Simple pixel drawing
      int w = goalGridSize;
      auto drawRect = [&](int x, int y, int rw, int rh) {
        for (int iy = y; iy < y + rh; iy++) {
          for (int ix = x; ix < x + rw; ix++) {
            if (ix >= 0 && ix < w && iy >= 0 && iy < w)
              data[iy * w + ix] = 1.0f;
          }
        }
      };

      int s = w / 10;  // scale
      int thick = s / 2;
      // H
      drawRect(2 * s, 3 * s, thick, 4 * s);
      drawRect(4 * s, 3 * s, thick, 4 * s);
      drawRect(2 * s, 5 * s, 2 * s + thick, thick);
      // I
      drawRect(6 * s, 3 * s, thick, 4 * s);
    } else if (params.goalMode == 4) {  // Image
      if (!loadBMP(params.goalImagePath, data, goalGridSize)) {
// Fallback X pattern
#pragma omp parallel for collapse(2)
        for (int y = 0; y < goalGridSize; y++) {
          for (int x = 0; x < goalGridSize; x++) {
            if (abs(x - y) < 20 || abs(x - (goalGridSize - y)) < 20)
              data[y * goalGridSize + x] = 1.0f;
          }
        }
      }
    }

    glBindTexture(GL_TEXTURE_2D, goalTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, goalGridSize, goalGridSize, GL_RED,
                    GL_FLOAT, data.data());
  }

  void saveScene(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to save scene: " << filename << std::endl;
        return;
    }
    
    out << "worldWidth=" << params.worldWidth << "\n";
    out << "worldHeight=" << params.worldHeight << "\n";
    out << "worldDepth=" << params.worldDepth << "\n";
    out << "numParticles=" << params.numParticles << "\n";
    out << "maxParticles=" << params.maxParticles << "\n";
    out << "w_k=" << params.w_k << "\n";
    out << "mu_k=" << params.mu_k << "\n";
    out << "sigma_k2=" << params.sigma_k2 << "\n";
    out << "mu_g=" << params.mu_g << "\n";
    out << "sigma_g2=" << params.sigma_g2 << "\n";
    out << "c_rep=" << params.c_rep << "\n";
    out << "dt=" << params.dt << "\n";
    out << "h=" << params.h << "\n";
    out << "evolutionEnabled=" << params.evolutionEnabled << "\n";
    out << "birthRate=" << params.birthRate << "\n";
    out << "deathRate=" << params.deathRate << "\n";
    out << "mutationRate=" << params.mutationRate << "\n";
    out << "energyDecay=" << params.energyDecay << "\n";
    out << "energyFromGrowth=" << params.energyFromGrowth << "\n";
    out << "translateX=" << params.translateX << "\n";
    out << "translateY=" << params.translateY << "\n";
    out << "translateZ=" << params.translateZ << "\n";
    out << "zoom=" << params.zoom << "\n";
    out << "stepsPerFrame=" << params.stepsPerFrame << "\n";
    out << "showFields=" << params.showFields << "\n";
    out << "fieldType=" << params.fieldType << "\n";
    out << "foodEnabled=" << params.foodEnabled << "\n";
    out << "foodSpawnRate=" << params.foodSpawnRate << "\n";
    out << "foodDecayRate=" << params.foodDecayRate << "\n";
    out << "foodMaxAmount=" << params.foodMaxAmount << "\n";
    out << "foodConsumptionRadius=" << params.foodConsumptionRadius << "\n";
    out << "showFood=" << params.showFood << "\n";
    out << "view3D=" << params.view3D << "\n";
    out << "cameraAngle=" << params.cameraAngle << "\n";
    out << "cameraRotation=" << params.cameraRotation << "\n";
    out << "cameraDistance=" << params.cameraDistance << "\n";
    out << "heightScale=" << params.heightScale << "\n";
    out << "glowIntensity=" << params.glowIntensity << "\n";
    out << "showWireframe=" << params.showWireframe << "\n";
    out << "ambientLight=" << params.ambientLight << "\n";
    out << "particleSize=" << params.particleSize << "\n";
    out << "interactionMode=" << params.interactionMode << "\n";
    out << "brushRadius=" << params.brushRadius << "\n";
    out << "forceStrength=" << params.forceStrength << "\n";
    out << "goalMode=" << params.goalMode << "\n";
    out << "goalStrength=" << params.goalStrength << "\n";
    out << "showGoal=" << params.showGoal << "\n";
    out << "goalImagePath=" << params.goalImagePath << "\n";
    
    std::cout << "Scene saved to " << filename << std::endl;
  }

  void loadScene(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Failed to load scene: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;
        
        std::string key = line.substr(0, eqPos);
        std::string val = line.substr(eqPos + 1);
        
        try {
            if (key == "worldWidth") params.worldWidth = std::stof(val);
            else if (key == "worldHeight") params.worldHeight = std::stof(val);
            else if (key == "worldDepth") params.worldDepth = std::stof(val);
            else if (key == "numParticles") params.numParticles = std::stoi(val);
            else if (key == "maxParticles") params.maxParticles = std::stoi(val);
            else if (key == "w_k") params.w_k = std::stof(val);
            else if (key == "mu_k") params.mu_k = std::stof(val);
            else if (key == "sigma_k2") params.sigma_k2 = std::stof(val);
            else if (key == "mu_g") params.mu_g = std::stof(val);
            else if (key == "sigma_g2") params.sigma_g2 = std::stof(val);
            else if (key == "c_rep") params.c_rep = std::stof(val);
            else if (key == "dt") params.dt = std::stof(val);
            else if (key == "h") params.h = std::stof(val);
            else if (key == "evolutionEnabled") params.evolutionEnabled = std::stoi(val);
            else if (key == "birthRate") params.birthRate = std::stof(val);
            else if (key == "deathRate") params.deathRate = std::stof(val);
            else if (key == "mutationRate") params.mutationRate = std::stof(val);
            else if (key == "energyDecay") params.energyDecay = std::stof(val);
            else if (key == "energyFromGrowth") params.energyFromGrowth = std::stof(val);
            else if (key == "translateX") params.translateX = std::stof(val);
            else if (key == "translateY") params.translateY = std::stof(val);
            else if (key == "translateZ") params.translateZ = std::stof(val);
            else if (key == "zoom") params.zoom = std::stof(val);
            else if (key == "stepsPerFrame") params.stepsPerFrame = std::stoi(val);
            else if (key == "showFields") params.showFields = std::stoi(val);
            else if (key == "fieldType") params.fieldType = std::stoi(val);
            else if (key == "foodEnabled") params.foodEnabled = std::stoi(val);
            else if (key == "foodSpawnRate") params.foodSpawnRate = std::stof(val);
            else if (key == "foodDecayRate") params.foodDecayRate = std::stof(val);
            else if (key == "foodMaxAmount") params.foodMaxAmount = std::stof(val);
            else if (key == "foodConsumptionRadius") params.foodConsumptionRadius = std::stof(val);
            else if (key == "showFood") params.showFood = std::stoi(val);
            else if (key == "view3D") params.view3D = std::stoi(val);
            else if (key == "cameraAngle") params.cameraAngle = std::stof(val);
            else if (key == "cameraRotation") params.cameraRotation = std::stof(val);
            else if (key == "cameraDistance") params.cameraDistance = std::stof(val);
            else if (key == "heightScale") params.heightScale = std::stof(val);
            else if (key == "glowIntensity") params.glowIntensity = std::stof(val);
            else if (key == "showWireframe") params.showWireframe = std::stoi(val);
            else if (key == "ambientLight") params.ambientLight = std::stof(val);
            else if (key == "particleSize") params.particleSize = std::stof(val);
            else if (key == "interactionMode") params.interactionMode = std::stoi(val);
            else if (key == "brushRadius") params.brushRadius = std::stof(val);
            else if (key == "forceStrength") params.forceStrength = std::stof(val);
            else if (key == "goalMode") params.goalMode = std::stoi(val);
            else if (key == "goalStrength") params.goalStrength = std::stof(val);
            else if (key == "showGoal") params.showGoal = std::stoi(val);
            else if (key == "goalImagePath") {
                if (val.length() < 256) strncpy(params.goalImagePath, val.c_str(), 255);
            }
        } catch (...) {
        }
    }
    std::cout << "Scene loaded from " << filename << std::endl;
    // Reinitialize to apply parameters
    init();
  }

  void initFood() {
    // Load food update shader
    foodUpdateShader = ComputeShader("shaders/food_update.comp");
    foodUpdateShader.init();

    // Create food texture (RGBA16F: R=food amount, G=freshness)
    glGenTextures(1, &foodTexture);
    glBindTexture(GL_TEXTURE_2D, foodTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, foodGridSize, foodGridSize, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Initialize food texture with some random food
    std::vector<float> foodData(foodGridSize * foodGridSize * 4, 0.0f);
    int totalCells = foodGridSize * foodGridSize;

#pragma omp parallel
    {
      std::mt19937 localRng(std::random_device{}() + omp_get_thread_num());
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);

#pragma omp for
      for (int i = 0; i < totalCells; i++) {
        if (dist(localRng) < 0.1f) {  // 10% initial food coverage
          foodData[i * 4 + 0] = dist(localRng) * 0.5f;  // Food amount
          foodData[i * 4 + 1] = 1.0f;                   // Freshness
        }
      }
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, foodGridSize, foodGridSize, GL_RGBA,
                    GL_FLOAT, foodData.data());
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void init3D() {
    // Load 3D shaders
    heightmapShader = ComputeShader("shaders/terrain_heightmap.comp");
    heightmapShader.init();

    terrainShader =
        RenderShader("shaders/terrain.vert", "shaders/terrain.frag");
    terrainShader.init();

    particle3DShader =
        RenderShader("shaders/particle3d.vert", "shaders/particle3d.frag");
    particle3DShader.init();

    // Create heightmap texture
    glGenTextures(1, &heightmapTexture);
    glBindTexture(GL_TEXTURE_2D, heightmapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, terrainGridSize, terrainGridSize,
                 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Create terrain mesh grid
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    for (int y = 0; y < terrainGridSize; y++) {
      for (int x = 0; x < terrainGridSize; x++) {
        float u = static_cast<float>(x) / (terrainGridSize - 1);
        float v = static_cast<float>(y) / (terrainGridSize - 1);
        vertices.push_back(u);
        vertices.push_back(v);
      }
    }

    for (int y = 0; y < terrainGridSize - 1; y++) {
      for (int x = 0; x < terrainGridSize - 1; x++) {
        int topLeft = y * terrainGridSize + x;
        int topRight = topLeft + 1;
        int bottomLeft = (y + 1) * terrainGridSize + x;
        int bottomRight = bottomLeft + 1;

        indices.push_back(topLeft);
        indices.push_back(bottomLeft);
        indices.push_back(topRight);
        indices.push_back(topRight);
        indices.push_back(bottomLeft);
        indices.push_back(bottomRight);
      }
    }

    terrainIndexCount = static_cast<int>(indices.size());

    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenBuffers(1, &terrainEBO);

    glBindVertexArray(terrainVAO);

    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                 indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Create empty VAO for particle rendering (uses gl_VertexID)
    glGenVertexArrays(1, &particleVAO);
  }

  void resetParticles() {
    // Spawn particles uniformly in the 3D world bounds
    std::vector<float> data(params.maxParticles * PARTICLE_FLOATS);

// Parallel initialization with thread-local RNGs
#pragma omp parallel
    {
      // Each thread gets its own RNG seeded uniquely
      std::mt19937 localRng(std::random_device{}() + omp_get_thread_num());
      std::uniform_real_distribution<float> posDistX(-params.worldWidth / 2.0f,
                                                     params.worldWidth / 2.0f);
      std::uniform_real_distribution<float> posDistY(-params.worldHeight / 2.0f,
                                                     params.worldHeight / 2.0f);
      std::uniform_real_distribution<float> posDistZ(-params.worldDepth / 2.0f,
                                                     params.worldDepth / 2.0f);
      std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
      std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);

#pragma omp for
      for (int i = 0; i < params.maxParticles; i++) {
        int base = i * PARTICLE_FLOATS;
        if (i < params.numParticles) {
          // Position (3D)
          data[base + 0] = posDistX(localRng);
          data[base + 1] = posDistY(localRng);
          data[base + 2] = posDistZ(localRng);
          // Velocity (3D)
          data[base + 3] = 0.0f;
          data[base + 4] = 0.0f;
          data[base + 5] = 0.0f;
          // Energy
          data[base + 6] = 1.0f;
          // Species
          data[base + 7] = speciesDist(localRng);
          // Age
          data[base + 8] = 0.0f;
          // DNA (5 values)
          for (int d = 0; d < 5; d++) {
            data[base + 9 + d] = dnaDist(localRng);
          }
        } else {
          // Dead/inactive particle (14 floats all zero)
          for (int j = 0; j < PARTICLE_FLOATS; j++) {
            data[base + j] = 0.0f;
          }
        }
      }
    }

    particleBufferA.setData(data);
    particleBufferB.setData(data);
    aliveCount = params.numParticles;
  }

  void step() {
    Buffer& readBuffer = useBufferA ? particleBufferA : particleBufferB;
    Buffer& writeBuffer = useBufferA ? particleBufferB : particleBufferA;

    // === STEP 1: Update food (spawn + decay) ===
    if (params.foodEnabled) {
      foodUpdateShader.use();

      // Bind food texture for read/write
      glBindImageTexture(0, foodTexture, 0, GL_FALSE, 0, GL_READ_WRITE,
                         GL_RGBA16F);

      // Set uniforms
      static int foodFrame = 0;
      foodUpdateShader.setUniform("u_FoodGridSize", foodGridSize);
      foodUpdateShader.setUniform("u_FoodSpawnRate", params.foodSpawnRate);
      foodUpdateShader.setUniform("u_FoodDecayRate", params.foodDecayRate);
      foodUpdateShader.setUniform("u_FoodMaxAmount", params.foodMaxAmount);
      foodUpdateShader.setUniform("u_RandomSeed", foodFrame++);

      // Dispatch (16x16 work groups)
      int foodWorkGroupsX = (foodGridSize + 15) / 16;
      int foodWorkGroupsY = (foodGridSize + 15) / 16;
      foodUpdateShader.dispatch(foodWorkGroupsX, foodWorkGroupsY, 1);
      foodUpdateShader.wait();
    }

    // === STEP 2: Update particles ===
    stepShader.use();

    // Bind buffers
    stepShader.bindBuffer("ParticlesIn", readBuffer, 0);
    stepShader.bindBuffer("ParticlesOut", writeBuffer, 1);

    // Bind food texture for read/write (particles consume food)
    if (params.foodEnabled) {
      glBindImageTexture(0, foodTexture, 0, GL_FALSE, 0, GL_READ_WRITE,
                         GL_RGBA16F);
    }

    // Bind Goal Texture
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, goalTexture);
    stepShader.setUniform("u_GoalTexture", 1);
    stepShader.setUniform("u_GoalMode", params.goalMode);
    stepShader.setUniform("u_GoalStrength", params.goalStrength);

    // Bind uniforms
    stepShader.setUniform("u_NumParticles", params.maxParticles);
    stepShader.setUniform("u_AliveCount", aliveCount);
    stepShader.setUniform("u_WorldWidth", params.worldWidth);
    stepShader.setUniform("u_WorldHeight", params.worldHeight);
    stepShader.setUniform("u_WorldDepth", params.worldDepth);
    stepShader.setUniform("u_Wk", params.w_k);
    stepShader.setUniform("u_MuK", params.mu_k);
    stepShader.setUniform("u_SigmaK2", params.sigma_k2);
    stepShader.setUniform("u_MuG", params.mu_g);
    stepShader.setUniform("u_SigmaG2", params.sigma_g2);
    stepShader.setUniform("u_Crep", params.c_rep);
    stepShader.setUniform("u_Dt", params.dt);
    stepShader.setUniform("u_H", params.h);
    stepShader.setUniform("u_EvolutionEnabled", params.evolutionEnabled);
    stepShader.setUniform("u_BirthRate", params.birthRate);
    stepShader.setUniform("u_DeathRate", params.deathRate);
    stepShader.setUniform("u_MutationRate", params.mutationRate);
    stepShader.setUniform("u_EnergyDecay", params.energyDecay);
    stepShader.setUniform("u_EnergyFromGrowth", params.energyFromGrowth);

    // Food system uniforms
    stepShader.setUniform("u_FoodGridSize", foodGridSize);
    stepShader.setUniform("u_FoodConsumptionRadius",
                          params.foodConsumptionRadius);

    // Random seed for evolution
    static int frame = 0;
    stepShader.setUniform("u_RandomSeed", frame++);

    // Dispatch compute shader (128 threads per workgroup)
    int workGroups = (params.maxParticles + 127) / 128;
    stepShader.dispatch(workGroups, 1, 1);
    stepShader.wait();

    useBufferA = !useBufferA;
  }

  void display(int windowWidth, int windowHeight) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;

    displayShader.use();
    displayShader.bindBuffer("Particles", activeBuffer, 0);

    // Bind food texture for display
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, foodTexture);
    displayShader.setUniform("u_FoodTexture", 0);

    displayShader.setUniform("u_NumParticles", params.maxParticles);
    displayShader.setUniform("u_WorldWidth", params.worldWidth);
    displayShader.setUniform("u_WorldHeight", params.worldHeight);
    displayShader.setUniform("u_TranslateX", params.translateX);
    displayShader.setUniform("u_TranslateY", params.translateY);
    displayShader.setUniform("u_Zoom", params.zoom);
    displayShader.setUniform("u_WindowWidth", static_cast<float>(windowWidth));
    displayShader.setUniform("u_WindowHeight",
                             static_cast<float>(windowHeight));
    displayShader.setUniform("u_Wk", params.w_k);
    displayShader.setUniform("u_MuK", params.mu_k);
    displayShader.setUniform("u_SigmaK2", params.sigma_k2);
    displayShader.setUniform("u_MuG", params.mu_g);
    displayShader.setUniform("u_SigmaG2", params.sigma_g2);
    displayShader.setUniform("u_ShowFields", params.showFields);
    displayShader.setUniform("u_FieldType", params.fieldType);
    displayShader.setUniform("u_ShowFood", params.showFood);
    displayShader.setUniform("u_FoodGridSize", foodGridSize);

    displayShader.render();
  }

  void display3D(int windowWidth, int windowHeight) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;

    // === Build camera matrices ===
    float aspect =
        static_cast<float>(windowWidth) / static_cast<float>(windowHeight);

    // Camera orbit around center
    float camRadius = params.cameraDistance;
    float camAngleRad = params.cameraAngle * 3.14159f / 180.0f;
    float camRotRad = params.cameraRotation * 3.14159f / 180.0f;

    float camX = camRadius * std::cos(camRotRad) * std::cos(camAngleRad);
    float camY = camRadius * std::sin(camAngleRad);
    float camZ = camRadius * std::sin(camRotRad) * std::cos(camAngleRad);

    // Simple view matrix (look at origin)
    float eye[3] = {camX, camY, camZ};
    float target[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};

    // Forward, right, up vectors
    float fwd[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
    float fwdLen =
        std::sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
    fwd[0] /= fwdLen;
    fwd[1] /= fwdLen;
    fwd[2] /= fwdLen;

    float right[3] = {fwd[1] * up[2] - fwd[2] * up[1],
                      fwd[2] * up[0] - fwd[0] * up[2],
                      fwd[0] * up[1] - fwd[1] * up[0]};
    float rightLen = std::sqrt(right[0] * right[0] + right[1] * right[1] +
                               right[2] * right[2]);
    right[0] /= rightLen;
    right[1] /= rightLen;
    right[2] /= rightLen;

    float upVec[3] = {right[1] * fwd[2] - right[2] * fwd[1],
                      right[2] * fwd[0] - right[0] * fwd[2],
                      right[0] * fwd[1] - right[1] * fwd[0]};

    // View matrix (column-major for OpenGL)
    float view[16] = {
        right[0],
        upVec[0],
        -fwd[0],
        0.0f,
        right[1],
        upVec[1],
        -fwd[1],
        0.0f,
        right[2],
        upVec[2],
        -fwd[2],
        0.0f,
        -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]),
        -(upVec[0] * eye[0] + upVec[1] * eye[1] + upVec[2] * eye[2]),
        (fwd[0] * eye[0] + fwd[1] * eye[1] + fwd[2] * eye[2]),
        1.0f};

    // Perspective projection
    float fov = 60.0f * 3.14159f / 180.0f;
    float nearPlane = 0.1f;
    float farPlane = 500.0f;
    float tanHalfFov = std::tan(fov / 2.0f);

    float proj[16] = {1.0f / (aspect * tanHalfFov),
                      0.0f,
                      0.0f,
                      0.0f,
                      0.0f,
                      1.0f / tanHalfFov,
                      0.0f,
                      0.0f,
                      0.0f,
                      0.0f,
                      -(farPlane + nearPlane) / (farPlane - nearPlane),
                      -1.0f,
                      0.0f,
                      0.0f,
                      -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane),
                      0.0f};

    // Multiply view * proj -> viewProj (manual matrix multiply)
    float viewProj[16];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        viewProj[i * 4 + j] = 0.0f;
        for (int k = 0; k < 4; k++) {
          viewProj[i * 4 + j] += proj[k * 4 + j] * view[i * 4 + k];
        }
      }
    }

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClear(GL_DEPTH_BUFFER_BIT);

    // === Render 3D particles ===
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // Additive blending for glow
    glEnable(GL_PROGRAM_POINT_SIZE);

    particle3DShader.use();
    particle3DShader.bindBuffer("Particles", activeBuffer, 0);

    particle3DShader.setUniformMat4("u_ViewProjection", viewProj);
    particle3DShader.setUniform("u_NumParticles", params.maxParticles);
    particle3DShader.setUniform("u_WorldWidth", params.worldWidth);
    particle3DShader.setUniform("u_WorldHeight", params.worldHeight);
    particle3DShader.setUniform("u_WorldDepth", params.worldDepth);
    particle3DShader.setUniform("u_ParticleSize", params.particleSize);
    particle3DShader.setUniform("u_TranslateX", params.translateX);
    particle3DShader.setUniform("u_TranslateY", params.translateY);
    particle3DShader.setUniform("u_TranslateZ", params.translateZ);
    particle3DShader.setUniform("u_Zoom", params.zoom);
    particle3DShader.setUniform("u_CameraPos", camX, camY, camZ);

    // Render particles as points
    glBindVertexArray(particleVAO);  // Empty VAO, using gl_VertexID
    glDrawArrays(GL_POINTS, 0, params.maxParticles);
    glBindVertexArray(0);

    glDisable(GL_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
  }

  void updateStats() {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
    std::vector<float> data = activeBuffer.getData();

    int localAliveCount = 0;
    float totalEnergy = 0.0f;
    float totalAge = 0.0f;

#pragma omp parallel for reduction(+ : localAliveCount, totalEnergy, totalAge)
    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * PARTICLE_FLOATS;
      float energy = data[base + 6];  // Energy is now at index 6
      float age = data[base + 8];     // Age is now at index 8

      if (energy > 0.01f) {
        localAliveCount++;
        totalEnergy += energy;
        totalAge += age;
      }
    }

    aliveCount = localAliveCount;
    avgEnergy = aliveCount > 0 ? totalEnergy / aliveCount : 0.0f;
    avgAge = aliveCount > 0 ? totalAge / aliveCount : 0.0f;
  }

  void addParticle(float x, float y, float z) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
    std::vector<float> data = activeBuffer.getData();

    // Find a dead slot
    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * PARTICLE_FLOATS;
      if (data[base + 6] < 0.01f) {  // Dead particle (energy at index 6)
        std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
        std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);

        data[base + 0] = x;                 // x
        data[base + 1] = y;                 // y
        data[base + 2] = z;                 // z
        data[base + 3] = 0.0f;              // vx
        data[base + 4] = 0.0f;              // vy
        data[base + 5] = 0.0f;              // vz
        data[base + 6] = 1.0f;              // energy
        data[base + 7] = speciesDist(rng);  // species
        data[base + 8] = 0.0f;              // age
        for (int d = 0; d < 5; d++) {
          data[base + 9 + d] = dnaDist(rng);
        }

        activeBuffer.setData(data);
        // Also update the other buffer
        Buffer& otherBuffer = useBufferA ? particleBufferB : particleBufferA;
        otherBuffer.setData(data);

        aliveCount++;
        break;
      }
    }
  }

  void spawnOrbium(float x, float y, float z) {
    // Spawn a cluster of particles that should form a soliton
    int count = 40;
    float radius = 3.0f;

    for (int i = 0; i < count; i++) {
      std::uniform_real_distribution<float> dist(-radius, radius);
      addParticle(x + dist(rng), y + dist(rng), z + dist(rng));
    }
  }

  void applyForce(float x, float y, float z, float strength, float radius) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
    std::vector<float> data = activeBuffer.getData();

    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * PARTICLE_FLOATS;
      if (data[base + 6] > 0.01f) {
        float dx = data[base + 0] - x;
        float dy = data[base + 1] - y;
        float dz = data[base + 2] - z;
        float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < radius * radius) {
          float dist = sqrt(dist2);
          float force = strength * (1.0f - dist / radius);
          if (dist > 0.001f) {
            data[base + 3] += (dx / dist) * force;
            data[base + 4] += (dy / dist) * force;
            data[base + 5] += (dz / dist) * force;
          }
        }
      }
    }
    activeBuffer.setData(data);
    Buffer& otherBuffer = useBufferA ? particleBufferB : particleBufferA;
    otherBuffer.setData(data);
  }
};

// Global simulation
ParticleLeniaSimulation simulation;
bool paused = false;

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
  WINDOW_WIDTH = width;
  WINDOW_HEIGHT = height;
}

void processInput(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }

  // Camera rotation with WASD
  float rotSpeed = 2.0f;
  float angleSpeed = 1.0f;

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    simulation.params.cameraAngle =
        std::min(89.0f, simulation.params.cameraAngle + angleSpeed);
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    simulation.params.cameraAngle =
        std::max(5.0f, simulation.params.cameraAngle - angleSpeed);
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    simulation.params.cameraRotation =
        std::fmod(simulation.params.cameraRotation - rotSpeed + 360.0f, 360.0f);
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    simulation.params.cameraRotation =
        std::fmod(simulation.params.cameraRotation + rotSpeed, 360.0f);
  }

  // Zoom with Q/E
  float zoomSpeed = 1.0f;
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
    simulation.params.cameraDistance =
        std::max(10.0f, simulation.params.cameraDistance - zoomSpeed);
  }
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
    simulation.params.cameraDistance =
        std::min(200.0f, simulation.params.cameraDistance + zoomSpeed);
  }
}

ImVec2 screenToWorld(float screenX, float screenY) {
  float windowAspect =
      static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);
  float worldAspect =
      simulation.params.worldWidth / simulation.params.worldHeight;

  // Normalized UV coordinates [-1, 1]
  float uvX = (screenX / WINDOW_WIDTH - 0.5f) * 2.0f;
  float uvY = ((1.0f - screenY / WINDOW_HEIGHT) - 0.5f) * 2.0f;

  // Apply aspect ratio correction (same as shader)
  if (windowAspect > worldAspect) {
    uvX *= windowAspect / worldAspect;
  } else {
    uvY *= worldAspect / windowAspect;
  }

  // UV is [-1, 1], multiply by half world size to get world coords
  float worldX =
      uvX * (simulation.params.worldWidth * 0.5f) / simulation.params.zoom +
      simulation.params.translateX;
  float worldY =
      uvY * (simulation.params.worldHeight * 0.5f) / simulation.params.zoom +
      simulation.params.translateY;
  return ImVec2(worldX, worldY);
}

void renderUI() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(320, 650), ImGuiCond_FirstUseEver);

  ImGui::Begin("Chronos Control Panel");

  // === SIMULATION STATUS ===
  ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "SIMULATION STATUS");
  ImGui::Text("Living Entities: %d / %d", simulation.aliveCount,
              simulation.params.maxParticles);
  ImGui::Text("Mean Vitality: %.2f", simulation.avgEnergy);
  ImGui::Text("Mean Lifespan: %.0f ticks", simulation.avgAge);
  ImGui::Text(
      "Performance: %.1f FPS (%.2f ms/tick)", ImGui::GetIO().Framerate,
      (1000.0f / ImGui::GetIO().Framerate) / simulation.params.stepsPerFrame);

  ImGui::Separator();

  // === PLAYBACK CONTROLS ===
  ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "PLAYBACK");

  float buttonWidth = 100.0f;
  if (ImGui::Button(paused ? "Resume" : "Halt", ImVec2(buttonWidth, 0))) {
    paused = !paused;
  }
  ImGui::SameLine();
  if (ImGui::Button("Reinitialize", ImVec2(buttonWidth, 0))) {
    simulation.resetParticles();
  }

  ImGui::SliderInt("Ticks per Frame", &simulation.params.stepsPerFrame, 1, 50);

  ImGui::Separator();

  // === SCENE MANAGEMENT ===
  ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "SCENE MANAGEMENT");
  static char sceneFilename[128] = "scene.txt";
  ImGui::InputText("Filename", sceneFilename, 128);
  if (ImGui::Button("Export Scene", ImVec2(buttonWidth, 0))) {
    simulation.saveScene(sceneFilename);
  }
  ImGui::SameLine();
  if (ImGui::Button("Import Scene", ImVec2(buttonWidth, 0))) {
    simulation.loadScene(sceneFilename);
  }

  ImGui::Separator();

  // === ENVIRONMENT ===
  if (ImGui::CollapsingHeader("Environment Settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::TextDisabled("World Dimensions (3D Cube)");
    ImGui::PushItemWidth(120);
    ImGui::DragFloat("Arena Width", &simulation.params.worldWidth, 0.5f, 10.0f,
                     100.0f);
    ImGui::DragFloat("Arena Height", &simulation.params.worldHeight, 0.5f,
                     10.0f, 100.0f);
    ImGui::DragFloat("Arena Depth", &simulation.params.worldDepth, 0.5f, 10.0f,
                     100.0f);
    ImGui::PopItemWidth();

    int numParticles = simulation.params.numParticles;
    if (ImGui::DragInt("Spawn Count", &numParticles, 5, 10, 1000)) {
      simulation.params.numParticles = numParticles;
    }
  }

  // === MOUSE INTERACTION ===
  if (ImGui::CollapsingHeader("Mouse Interaction", ImGuiTreeNodeFlags_DefaultOpen)) {
      const char* modes[] = { "Navigation Only", "Paint Particles", "Repel Force", "Attract Force", "Spawn Orbium" };
      ImGui::Combo("Active Tool", &simulation.params.interactionMode, modes, 5);
      
      if (simulation.params.interactionMode > 0) {
          ImGui::Indent();
          ImGui::DragFloat("Brush Radius", &simulation.params.brushRadius, 0.5f, 1.0f, 50.0f);
          if (simulation.params.interactionMode == 2 || simulation.params.interactionMode == 3) {
             ImGui::DragFloat("Force Strength", &simulation.params.forceStrength, 0.01f, 0.0f, 5.0f);
          }
          ImGui::Unindent();
          ImGui::TextColored(ImVec4(1,1,0,1), "Hold Left Click to use tool");
      }
  }

  // === INTERACTION PHYSICS ===
  if (ImGui::CollapsingHeader("Interaction Physics")) {
    ImGui::TextDisabled("Perception Kernel");
    ImGui::DragFloat("Sensitivity##k", &simulation.params.w_k, 0.001f, 0.001f,
                     0.1f, "%.4f");
    ImGui::DragFloat("Optimal Range##k", &simulation.params.mu_k, 0.1f, 0.5f,
                     20.0f);
    ImGui::DragFloat("Range Variance##k", &simulation.params.sigma_k2, 0.05f,
                     0.1f, 10.0f);

    ImGui::Spacing();
    ImGui::TextDisabled("Separation Force");
    ImGui::DragFloat("Push Intensity", &simulation.params.c_rep, 0.1f, 0.0f,
                     5.0f);
  }

  // === LENIA DYNAMICS ===
  if (ImGui::CollapsingHeader("Lenia Dynamics",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::TextDisabled("Growth Response Curve");
    ImGui::DragFloat("Target Density", &simulation.params.mu_g, 0.01f, 0.0f,
                     2.0f);
    ImGui::DragFloat("Density Tolerance", &simulation.params.sigma_g2, 0.001f,
                     0.001f, 0.5f, "%.4f");
  }

  // === TEMPORAL ===
  if (ImGui::CollapsingHeader("Temporal Settings")) {
    ImGui::DragFloat("Integration Step", &simulation.params.dt, 0.01f, 0.01f,
                     0.5f);
    ImGui::DragFloat("Gradient Epsilon", &simulation.params.h, 0.001f, 0.001f,
                     0.1f, "%.4f");
  }

  // === EVOLUTIONARY MECHANICS ===
  if (ImGui::CollapsingHeader("Evolutionary Mechanics",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Active Evolution", &simulation.params.evolutionEnabled);

    if (simulation.params.evolutionEnabled) {
      ImGui::Spacing();
      ImGui::TextDisabled("Population Dynamics");
      ImGui::DragFloat("Reproduction Chance", &simulation.params.birthRate,
                       0.0001f, 0.0f, 0.01f, "%.5f");
      ImGui::DragFloat("Mortality Baseline", &simulation.params.deathRate,
                       0.0001f, 0.0f, 0.01f, "%.5f");

      ImGui::Spacing();
      ImGui::TextDisabled("Genetics");
      ImGui::DragFloat("Mutation Amplitude", &simulation.params.mutationRate,
                       0.01f, 0.0f, 0.5f);

      ImGui::Spacing();
      ImGui::TextDisabled("Metabolism");
      ImGui::DragFloat("Vitality Drain", &simulation.params.energyDecay,
                       0.0001f, 0.0f, 0.01f, "%.5f");
      ImGui::DragFloat("Vitality Gain", &simulation.params.energyFromGrowth,
                       0.001f, 0.0f, 0.1f);
    }
  }

  // === FOOD SYSTEM ===
  if (ImGui::CollapsingHeader("Food System", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Enable Food", &simulation.params.foodEnabled);

    if (simulation.params.foodEnabled) {
      ImGui::Checkbox("Show Food", &simulation.params.showFood);

      ImGui::Spacing();
      ImGui::TextDisabled("Food Dynamics");
      ImGui::DragFloat("Spawn Rate", &simulation.params.foodSpawnRate, 0.0001f,
                       0.0f, 0.01f, "%.4f");
      ImGui::DragFloat("Decay Rate", &simulation.params.foodDecayRate, 0.0001f,
                       0.0f, 0.01f, "%.4f");
      ImGui::DragFloat("Max Amount", &simulation.params.foodMaxAmount, 0.1f,
                       0.1f, 5.0f);
      ImGui::DragFloat("Consumption Radius",
                       &simulation.params.foodConsumptionRadius, 0.1f, 0.5f,
                       10.0f);
    }
  }

  // === GOAL SYSTEM ===
  if (ImGui::CollapsingHeader("Goal Seeking", ImGuiTreeNodeFlags_DefaultOpen)) {
    bool changed = false;
    const char* goalModes[] = {"None", "Circle", "Box", "Text 'HI'",
                               "Image (BMP)"};
    if (ImGui::Combo("Goal Pattern", &simulation.params.goalMode, goalModes,
                     5)) {
      changed = true;
    }

    if (simulation.params.goalMode == 4) {
      if (ImGui::InputText("BMP Filename", simulation.params.goalImagePath,
                           256)) {
        // Delay update until button press or Enter?
        // InputText returns true on Enter or lost focus with change
      }
      if (ImGui::Button("Reload Image")) {
        changed = true;
      }
      ImGui::SameLine();
      ImGui::TextDisabled("(Supports 24-bit .bmp)");
    }

    ImGui::DragFloat("Attraction Strength", &simulation.params.goalStrength,
                     0.01f, 0.0f, 2.0f);

    if (changed) {
      simulation.updateGoalTexture();
    }
  }

  // === VISUALIZATION ===
  if (ImGui::CollapsingHeader("Visualization",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("3D View", &simulation.params.view3D);

    if (simulation.params.view3D) {
      ImGui::Indent();
      ImGui::TextDisabled("Camera Controls");
      ImGui::DragFloat("Camera Angle", &simulation.params.cameraAngle, 1.0f,
                       5.0f, 89.0f);
      ImGui::DragFloat("Camera Rotation", &simulation.params.cameraRotation,
                       2.0f, 0.0f, 360.0f);
      ImGui::DragFloat("Camera Distance", &simulation.params.cameraDistance,
                       1.0f, 10.0f, 200.0f);

      ImGui::Spacing();
      ImGui::TextDisabled("Particles");
      ImGui::DragFloat("Particle Size", &simulation.params.particleSize, 1.0f,
                       1.0f, 50.0f);
      ImGui::DragFloat("Glow Intensity", &simulation.params.glowIntensity, 0.1f,
                       0.0f, 3.0f);
      ImGui::Unindent();
    } else {
      ImGui::Checkbox("Render Field Overlay", &simulation.params.showFields);
      const char* fieldModes[] = {"Off", "Density Field", "Separation Field",
                                  "Growth Field", "Energy Landscape"};
      ImGui::Combo("Field Mode", &simulation.params.fieldType, fieldModes, 5);
    }

    ImGui::DragFloat("View Scale", &simulation.params.zoom, 0.05f, 0.1f, 5.0f);
  }

  ImGui::Separator();
  ImGui::TextDisabled("Camera: WASD = Rotate | Q/E = Zoom | Scroll = Zoom");

  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main() {
  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  // Fixed window size

  GLFWwindow* window =
      glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                       "Chronos - Particle Lenia Evolution", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSwapInterval(1);  // VSync

  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // Initialize ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450");

  // Initialize simulation
  simulation.init();
  simulation.init3D();

  // Pan tracking
  static ImVec2 panStart;
  static float panStartX, panStartY;

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    processInput(window);
    glfwPollEvents();

    // Handle input
    if (!io.WantCaptureMouse) {
      // Pan with middle mouse
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
        panStart = ImGui::GetMousePos();
        panStartX = simulation.params.translateX;
        panStartY = simulation.params.translateY;
      }
      if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        ImVec2 pos = ImGui::GetMousePos();
      
      // === INTERACTION TOOLS ===
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          ImVec2 mousePos = ImGui::GetMousePos();
          ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
          
          if (simulation.params.interactionMode == 1) { // Paint Particles
              // Spawn multiple particles per frame for "brush" feel
              for(int k=0; k<5; k++) {
                  std::uniform_real_distribution<float> dist(-simulation.params.brushRadius, simulation.params.brushRadius);
                  float rx = dist(simulation.rng);
                  float ry = dist(simulation.rng);
                  // Circular brush
                  if (rx*rx + ry*ry <= simulation.params.brushRadius*simulation.params.brushRadius) {
                      // Spread in Z slightly to avoid 2D planarity issues if in 3D
                      float rz = dist(simulation.rng) * 0.1f; 
                      simulation.addParticle(worldPos.x + rx, worldPos.y + ry, rz);
                  }
              }
          }
          else if (simulation.params.interactionMode == 2) { // Repel
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, simulation.params.forceStrength, simulation.params.brushRadius);
          }
          else if (simulation.params.interactionMode == 3) { // Attract
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, -simulation.params.forceStrength, simulation.params.brushRadius);
          }
          else if (simulation.params.interactionMode == 4) { // Spawn Orbium
              // Single click trigger check handled by IsMouseClicked
          }
      }
      
      // Separate check for single-click actions (Orbium)
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
           ImVec2 mousePos = ImGui::GetMousePos();
           ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
           
           if (simulation.params.interactionMode == 4) { // Spawn Orbium
               simulation.spawnOrbium(worldPos.x, worldPos.y, 0.0f);
           }
      }
        float dx = (pos.x - panStart.x) / WINDOW_WIDTH *
                   simulation.params.worldWidth * 2.0f / simulation.params.zoom;
        float dy = (pos.y - panStart.y) / WINDOW_HEIGHT *
                   simulation.params.worldHeight * 2.0f /
                   simulation.params.zoom;
        simulation.params.translateX = panStartX - dx;
        simulation.params.translateY = panStartY + dy;
      }

      // Zoom with scroll
      float scroll = io.MouseWheel;
      if (scroll != 0) {
        simulation.params.zoom *= (1.0f + scroll * 0.1f);
        simulation.params.zoom =
            std::max(0.1f, std::min(10.0f, simulation.params.zoom));
      }
      
      // === INTERACTION TOOLS ===
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          ImVec2 mousePos = ImGui::GetMousePos();
          ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
          
          if (simulation.params.interactionMode == 1) { // Paint Particles
              for(int k=0; k<5; k++) {
                  std::uniform_real_distribution<float> dist(-simulation.params.brushRadius, simulation.params.brushRadius);
                  float rx = dist(simulation.rng);
                  float ry = dist(simulation.rng);
                  if (rx*rx + ry*ry <= simulation.params.brushRadius*simulation.params.brushRadius) {
                      float rz = dist(simulation.rng) * 0.1f;
                      simulation.addParticle(worldPos.x + rx, worldPos.y + ry, rz);
                  }
              }
          }
          else if (simulation.params.interactionMode == 2) { // Repel
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, simulation.params.forceStrength, simulation.params.brushRadius);
          }
          else if (simulation.params.interactionMode == 3) { // Attract
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, -simulation.params.forceStrength, simulation.params.brushRadius);
          }
      }
      
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
           ImVec2 mousePos = ImGui::GetMousePos();
           ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
           if (simulation.params.interactionMode == 4) { // Spawn Orbium
               simulation.spawnOrbium(worldPos.x, worldPos.y, 0.0f);
           }
      }
    }

    // Update simulation
    if (!paused) {
      for (int i = 0; i < simulation.params.stepsPerFrame; i++) {
        simulation.step();
      }

      // Update stats every 10 frames
      static int frameCount = 0;
      if (++frameCount % 10 == 0) {
        simulation.updateStats();
      }
    }

    // Render
    if (simulation.params.view3D) {
      // Slightly brighter background for 3D to show depth
      glClearColor(0.01f, 0.03f, 0.06f, 1.0f);
    } else {
      glClearColor(0.0f, 0.02f, 0.05f, 1.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (simulation.params.view3D) {
      simulation.display3D(WINDOW_WIDTH, WINDOW_HEIGHT);
    } else {
      simulation.display(WINDOW_WIDTH, WINDOW_HEIGHT);
    }

    renderUI();

    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
