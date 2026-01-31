#pragma once

#include <glm/glm.hpp>
#include <random>
#include <vector>

/**
 * ParticleLife3D - A 3D implementation of "Particle Life" / "Primordial Soup"
 *
 * Particles of different colors attract/repel each other based on a rule matrix,
 * creating emergent lifelike behavior in 3D space.
 */
class ParticleLife3D {
 public:
  static constexpr int NUM_COLORS = 6;
  static constexpr int DEFAULT_PARTICLE_COUNT = 8000;

  struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    int colorIndex;
  };

  ParticleLife3D();
  ~ParticleLife3D();

  void init(int particleCount = DEFAULT_PARTICLE_COUNT);
  void update(float deltaTime);
  void randomizeRules();

  // Accessors
  std::vector<Particle> const& getParticles() const { return _particles; }
  int getParticleCount() const { return static_cast<int>(_particles.size()); }
  
  // Get color for a color index
  glm::vec3 getColor(int colorIndex) const;

  // Rule matrix access (for UI)
  float getRule(int colorA, int colorB) const { return _rules[colorA][colorB]; }
  void setRule(int colorA, int colorB, float value) { _rules[colorA][colorB] = value; }

  // Parameters
  float friction = 0.1f;
  float maxSpeed = 2.0f;
  float interactionRadius = 0.3f;
  float forceStrength = 0.5f;
  float worldSize = 3.0f;
  bool wrapEdges = true;

 private:
  void applyForces(float deltaTime);
  void updatePositions(float deltaTime);
  float attractionForce(float distance, float attraction) const;

  std::vector<Particle> _particles;
  
  // Rule matrix: rules[i][j] = how color i feels about color j
  // Positive = attraction, Negative = repulsion
  float _rules[NUM_COLORS][NUM_COLORS];

  // Predefined vibrant colors
  glm::vec3 _colors[NUM_COLORS] = {
      {1.0f, 0.2f, 0.2f},   // Red
      {0.2f, 1.0f, 0.3f},   // Green
      {0.2f, 0.4f, 1.0f},   // Blue
      {1.0f, 1.0f, 0.2f},   // Yellow
      {1.0f, 0.2f, 1.0f},   // Magenta
      {0.2f, 1.0f, 1.0f},   // Cyan
  };

  std::mt19937 _rng;
};
