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

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "core/Buffer.h"
#include "core/ComputeShader.h"
#include "core/RenderShader.h"

// Window dimensions
int WINDOW_WIDTH = 1200;
int WINDOW_HEIGHT = 900;

// Simulation parameters
struct SimulationParams {
  // World dimensions
  float worldWidth = 40.0f;
  float worldHeight = 40.0f;

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
  float zoom = 1.0f;

  // Rendering
  int stepsPerFrame = 5;
  bool showFields = true;
  int fieldType = 3;  // 0=none, 1=U, 2=R, 3=G, 4=E

  // 3D Rendering
  bool view3D = false;
  float cameraAngle = 45.0f;      // Degrees from horizontal
  float cameraRotation = 0.0f;    // Rotation around Y axis
  float cameraDistance = 60.0f;   // Distance from center
  float heightScale = 10.0f;      // Height multiplier for terrain
  float glowIntensity = 1.5f;     // Glow effect strength
  bool showWireframe = false;     // Show terrain wireframe
  float ambientLight = 0.5f;      // Ambient lighting (higher for visibility)
  float particleSize = 20.0f;     // 3D particle size
};

// Particle structure (must match shader)
struct Particle {
  float x, y;     // Position
  float vx, vy;   // Velocity
  float energy;   // Health/energy [0, 1]
  float species;  // Species ID (affects color)
  float age;      // Age in simulation steps
  float dna[5];   // Genetic parameters (mu_k, sigma_k2, mu_g, sigma_g2, c_rep
                  // variations)
};

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

  std::mt19937 rng;

  // Stats
  int aliveCount = 0;
  float avgEnergy = 0.0f;
  float avgAge = 0.0f;

  void init() {
    // Initialize RNG
    rng = std::mt19937(std::random_device{}());

    // Calculate buffer size: each particle has 12 floats
    int particleFloats = 12;
    int bufferSize = params.maxParticles * particleFloats;

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
  }

  void init3D() {
    // Load 3D shaders
    heightmapShader = ComputeShader("shaders/terrain_heightmap.comp");
    heightmapShader.init();

    terrainShader = RenderShader("shaders/terrain.vert", "shaders/terrain.frag");
    terrainShader.init();

    particle3DShader = RenderShader("shaders/particle3d.vert", "shaders/particle3d.frag");
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

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Create empty VAO for particle rendering (uses gl_VertexID)
    glGenVertexArrays(1, &particleVAO);
  }

  void resetParticles() {
    // Spawn particles uniformly in the world bounds
    std::uniform_real_distribution<float> posDistX(-params.worldWidth / 2.0f,
                                                   params.worldWidth / 2.0f);
    std::uniform_real_distribution<float> posDistY(-params.worldHeight / 2.0f,
                                                   params.worldHeight / 2.0f);
    std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
    std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);

    std::vector<float> data;
    data.reserve(params.maxParticles * 12);

    for (int i = 0; i < params.maxParticles; i++) {
      if (i < params.numParticles) {
        // Position
        data.push_back(posDistX(rng));
        data.push_back(posDistY(rng));
        // Velocity
        data.push_back(0.0f);
        data.push_back(0.0f);
        // Energy
        data.push_back(1.0f);
        // Species
        data.push_back(speciesDist(rng));
        // Age
        data.push_back(0.0f);
        // DNA (5 values)
        for (int d = 0; d < 5; d++) {
          data.push_back(dnaDist(rng));
        }
      } else {
        // Dead/inactive particle
        data.push_back(0.0f);  // x
        data.push_back(0.0f);  // y
        data.push_back(0.0f);  // vx
        data.push_back(0.0f);  // vy
        data.push_back(0.0f);  // energy = 0 means dead
        data.push_back(0.0f);  // species
        data.push_back(0.0f);  // age
        for (int d = 0; d < 5; d++) {
          data.push_back(0.0f);
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

    stepShader.use();

    // Bind buffers
    stepShader.bindBuffer("ParticlesIn", readBuffer, 0);
    stepShader.bindBuffer("ParticlesOut", writeBuffer, 1);

    // Bind uniforms
    stepShader.setUniform("u_NumParticles", params.maxParticles);
    stepShader.setUniform("u_AliveCount", aliveCount);
    stepShader.setUniform("u_WorldWidth", params.worldWidth);
    stepShader.setUniform("u_WorldHeight", params.worldHeight);
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

    displayShader.render();
  }

  void display3D(int windowWidth, int windowHeight) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;

    // === STEP 1: Generate heightmap from particles ===
    heightmapShader.use();
    
    // Bind particle buffer
    heightmapShader.bindBuffer("Particles", activeBuffer, 0);
    
    // Bind heightmap as image for writing
    glBindImageTexture(0, heightmapTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    
    heightmapShader.setUniform("u_HeightmapSize", terrainGridSize);
    heightmapShader.setUniform("u_NumParticles", params.maxParticles);
    heightmapShader.setUniform("u_WorldWidth", params.worldWidth);
    heightmapShader.setUniform("u_WorldHeight", params.worldHeight);
    heightmapShader.setUniform("u_Wk", params.w_k);
    heightmapShader.setUniform("u_MuK", params.mu_k);
    heightmapShader.setUniform("u_SigmaK2", params.sigma_k2);
    
    // Dispatch compute shader (one thread per heightmap texel)
    int workGroupsX = (terrainGridSize + 15) / 16;
    int workGroupsY = (terrainGridSize + 15) / 16;
    heightmapShader.dispatch(workGroupsX, workGroupsY, 1);
    heightmapShader.wait();

    // === STEP 2: Build camera matrices ===
    float aspect = static_cast<float>(windowWidth) / static_cast<float>(windowHeight);
    
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
    float fwdLen = std::sqrt(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    fwd[0] /= fwdLen; fwd[1] /= fwdLen; fwd[2] /= fwdLen;
    
    float right[3] = {
      fwd[1] * up[2] - fwd[2] * up[1],
      fwd[2] * up[0] - fwd[0] * up[2],
      fwd[0] * up[1] - fwd[1] * up[0]
    };
    float rightLen = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    right[0] /= rightLen; right[1] /= rightLen; right[2] /= rightLen;
    
    float upVec[3] = {
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0]
    };
    
    // View matrix (column-major for OpenGL)
    float view[16] = {
      right[0], upVec[0], -fwd[0], 0.0f,
      right[1], upVec[1], -fwd[1], 0.0f,
      right[2], upVec[2], -fwd[2], 0.0f,
      -(right[0]*eye[0] + right[1]*eye[1] + right[2]*eye[2]),
      -(upVec[0]*eye[0] + upVec[1]*eye[1] + upVec[2]*eye[2]),
      (fwd[0]*eye[0] + fwd[1]*eye[1] + fwd[2]*eye[2]),
      1.0f
    };
    
    // Perspective projection
    float fov = 60.0f * 3.14159f / 180.0f;
    float nearPlane = 0.1f;
    float farPlane = 500.0f;
    float tanHalfFov = std::tan(fov / 2.0f);
    
    float proj[16] = {
      1.0f / (aspect * tanHalfFov), 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f / tanHalfFov, 0.0f, 0.0f,
      0.0f, 0.0f, -(farPlane + nearPlane) / (farPlane - nearPlane), -1.0f,
      0.0f, 0.0f, -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane), 0.0f
    };
    
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

    // === STEP 3: Render terrain ===
    terrainShader.use();
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, heightmapTexture);
    terrainShader.setUniform("u_Heightmap", 0);
    
    terrainShader.setUniformMat4("u_ViewProjection", viewProj);
    terrainShader.setUniform("u_WorldWidth", params.worldWidth);
    terrainShader.setUniform("u_WorldHeight", params.worldHeight);
    terrainShader.setUniform("u_MaxHeight", params.heightScale);
    terrainShader.setUniform("u_TranslateX", params.translateX);
    terrainShader.setUniform("u_TranslateY", params.translateY);
    terrainShader.setUniform("u_Zoom", params.zoom);
    terrainShader.setUniform("u_CameraPos", camX, camY, camZ);
    terrainShader.setUniform("u_GlowIntensity", params.glowIntensity);
    terrainShader.setUniform("u_AmbientLight", params.ambientLight);
    terrainShader.setUniform("u_Wireframe", params.showWireframe ? 1 : 0);
    
    // Draw terrain mesh
    glBindVertexArray(terrainVAO);
    if (params.showWireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    glDrawElements(GL_TRIANGLES, terrainIndexCount, GL_UNSIGNED_INT, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindVertexArray(0);

    // === STEP 4: Render 3D particles ===
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // Additive blending for glow
    glEnable(GL_PROGRAM_POINT_SIZE);

    particle3DShader.use();
    particle3DShader.bindBuffer("Particles", activeBuffer, 0);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, heightmapTexture);
    particle3DShader.setUniform("u_Heightmap", 0);
    
    particle3DShader.setUniformMat4("u_ViewProjection", viewProj);
    particle3DShader.setUniform("u_NumParticles", params.maxParticles);
    particle3DShader.setUniform("u_WorldWidth", params.worldWidth);
    particle3DShader.setUniform("u_WorldHeight", params.worldHeight);
    particle3DShader.setUniform("u_MaxHeight", params.heightScale);
    particle3DShader.setUniform("u_ParticleSize", params.particleSize);
    particle3DShader.setUniform("u_TranslateX", params.translateX);
    particle3DShader.setUniform("u_TranslateY", params.translateY);
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

    aliveCount = 0;
    float totalEnergy = 0.0f;
    float totalAge = 0.0f;

    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * 12;
      float energy = data[base + 4];
      float age = data[base + 6];

      if (energy > 0.01f) {
        aliveCount++;
        totalEnergy += energy;
        totalAge += age;
      }
    }

    avgEnergy = aliveCount > 0 ? totalEnergy / aliveCount : 0.0f;
    avgAge = aliveCount > 0 ? totalAge / aliveCount : 0.0f;
  }

  void addParticle(float x, float y) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
    std::vector<float> data = activeBuffer.getData();

    // Find a dead slot
    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * 12;
      if (data[base + 4] < 0.01f) {  // Dead particle
        std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
        std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);

        data[base + 0] = x;                 // x
        data[base + 1] = y;                 // y
        data[base + 2] = 0.0f;              // vx
        data[base + 3] = 0.0f;              // vy
        data[base + 4] = 1.0f;              // energy
        data[base + 5] = speciesDist(rng);  // species
        data[base + 6] = 0.0f;              // age
        for (int d = 0; d < 5; d++) {
          data[base + 7 + d] = dnaDist(rng);
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

  // === ENVIRONMENT ===
  if (ImGui::CollapsingHeader("Environment Settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::TextDisabled("World Dimensions");
    ImGui::PushItemWidth(120);
    ImGui::DragFloat("Arena Width", &simulation.params.worldWidth, 0.5f, 10.0f,
                     100.0f);
    ImGui::DragFloat("Arena Height", &simulation.params.worldHeight, 0.5f,
                     10.0f, 100.0f);
    ImGui::PopItemWidth();

    int numParticles = simulation.params.numParticles;
    if (ImGui::DragInt("Spawn Count", &numParticles, 5, 10, 1000)) {
      simulation.params.numParticles = numParticles;
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

  // === VISUALIZATION ===
  if (ImGui::CollapsingHeader("Visualization")) {
    ImGui::Checkbox("3D View", &simulation.params.view3D);
    
    if (simulation.params.view3D) {
      ImGui::Indent();
      ImGui::TextDisabled("Camera Controls");
      ImGui::DragFloat("Camera Angle", &simulation.params.cameraAngle, 1.0f, 5.0f, 89.0f);
      ImGui::DragFloat("Camera Rotation", &simulation.params.cameraRotation, 2.0f, 0.0f, 360.0f);
      ImGui::DragFloat("Camera Distance", &simulation.params.cameraDistance, 1.0f, 10.0f, 200.0f);
      
      ImGui::Spacing();
      ImGui::TextDisabled("Terrain Effects");
      ImGui::DragFloat("Height Scale", &simulation.params.heightScale, 0.5f, 1.0f, 50.0f);
      ImGui::DragFloat("Glow Intensity", &simulation.params.glowIntensity, 0.1f, 0.0f, 3.0f);
      ImGui::DragFloat("Ambient Light", &simulation.params.ambientLight, 0.02f, 0.0f, 1.0f);
      ImGui::Checkbox("Wireframe Mode", &simulation.params.showWireframe);
      
      ImGui::Spacing();
      ImGui::TextDisabled("Particles");
      ImGui::DragFloat("Particle Size", &simulation.params.particleSize, 0.5f, 1.0f, 20.0f);
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
  ImGui::TextDisabled(
      "Keybinds: [A]+Mouse = Spawn | MMB = Pan | Scroll = Zoom");

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
      // Add particles with 'A' key
      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        ImVec2 worldPos = screenToWorld(mx, my);
        simulation.addParticle(worldPos.x, worldPos.y);
      }

      // Pan with middle mouse
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
        panStart = ImGui::GetMousePos();
        panStartX = simulation.params.translateX;
        panStartY = simulation.params.translateY;
      }
      if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        ImVec2 pos = ImGui::GetMousePos();
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
