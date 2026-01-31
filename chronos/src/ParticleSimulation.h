#pragma once

#include <random>
#include <vector>

/**
 * ParticleSimulation - A simple GPU-accelerated particle simulation.
 *
 * Creates visually interesting emergent patterns that look great
 * when stacked in 3D time.
 */
class ParticleSimulation {
 public:
  ParticleSimulation(int width, int height);
  ~ParticleSimulation();

  void update(float deltaTime);
  void renderPreview();

  unsigned int getOutputTexture() const { return _outputTexture; }
  unsigned int getOutputFBO() const { return _fbo; }

  // Simulation parameters
  void setParticleCount(int count);
  int getParticleCount() const { return _particleCount; }

 private:
  void initParticles();
  void initGL();
  void renderToTexture();

  struct Particle {
    float x, y;
    float vx, vy;
    float r, g, b;
    float life;
    float size;
  };

  int _width;
  int _height;
  int _particleCount = 5000;

  std::vector<Particle> _particles;
  std::mt19937 _rng;

  // OpenGL resources
  unsigned int _fbo = 0;
  unsigned int _outputTexture = 0;
  unsigned int _shaderProgram = 0;
  unsigned int _vao = 0;
  unsigned int _vbo = 0;
  unsigned int _previewVao = 0;
  unsigned int _previewVbo = 0;
  unsigned int _previewShader = 0;

  // Attractor points for emergent behavior
  struct Attractor {
    float x, y;
    float strength;
    float phase;
  };
  std::vector<Attractor> _attractors;
  float _time = 0.0f;
};
