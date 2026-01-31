#pragma once

#include <glm/glm.hpp>
#include <vector>

class ParticleLifeCUDA;
struct ParticleCUDA;

/**
 * ParticleRenderer3D - Renders 3D particles as glowing spheres/points
 */
class ParticleRenderer3D {
 public:
  ParticleRenderer3D();
  ~ParticleRenderer3D();

  void init();
  void shutdown();
              
  // CUDA version
  void renderCUDA(const std::vector<ParticleCUDA>& particles, glm::mat4 const& view,
              glm::mat4 const& projection, float pointSize = 8.0f,
              float glowIntensity = 1.0f);

 private:
  unsigned int _shaderProgram = 0;
  unsigned int _vao = 0;
  unsigned int _vbo = 0;

  // Uniform locations
  int _locView = -1;
  int _locProjection = -1;
  int _locPointSize = -1;
  int _locGlowIntensity = -1;
};
