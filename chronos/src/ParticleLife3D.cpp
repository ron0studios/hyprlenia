#include "ParticleLife3D.h"

#include <algorithm>
#include <cmath>

ParticleLife3D::ParticleLife3D() : _rng(std::random_device{}()) {
  randomizeRules();
}

ParticleLife3D::~ParticleLife3D() = default;

void ParticleLife3D::init(int particleCount) {
  _particles.clear();
  _particles.reserve(particleCount);

  std::uniform_real_distribution<float> posDist(-worldSize * 0.5f,
                                                 worldSize * 0.5f);
  std::uniform_int_distribution<int> colorDist(0, NUM_COLORS - 1);

  for (int i = 0; i < particleCount; ++i) {
    Particle p;
    p.position = glm::vec3(posDist(_rng), posDist(_rng), posDist(_rng));
    p.velocity = glm::vec3(0.0f);
    p.colorIndex = colorDist(_rng);
    _particles.push_back(p);
  }
}

void ParticleLife3D::randomizeRules() {
  std::uniform_real_distribution<float> ruleDist(-1.0f, 1.0f);

  for (int i = 0; i < NUM_COLORS; ++i) {
    for (int j = 0; j < NUM_COLORS; ++j) {
      _rules[i][j] = ruleDist(_rng);
    }
  }
}

void ParticleLife3D::update(float deltaTime) {
  // Clamp delta time to avoid instability
  deltaTime = std::min(deltaTime, 0.033f);

  applyForces(deltaTime);
  updatePositions(deltaTime);
}

float ParticleLife3D::attractionForce(float distance, float attraction) const {
  // Particle life force curve: repel when very close, then attract/repel based
  // on rule
  const float minDist = 0.02f;
  const float beta = 0.3f;  // Where attraction starts

  if (distance < minDist) {
    return 0.0f;  // Too close, ignore
  }

  if (distance < beta) {
    // Repulsion zone (universal) - particles don't overlap
    return (distance / beta - 1.0f);
  }

  if (distance < 1.0f) {
    // Attraction/repulsion zone based on rule
    return attraction * (1.0f - std::abs(2.0f * distance - 1.0f - beta) / (1.0f - beta));
  }

  return 0.0f;
}

void ParticleLife3D::applyForces(float deltaTime) {
  const float radiusSq = interactionRadius * interactionRadius;

  // O(n²) but simple - could optimize with spatial hashing
  for (size_t i = 0; i < _particles.size(); ++i) {
    glm::vec3 totalForce(0.0f);

    for (size_t j = 0; j < _particles.size(); ++j) {
      if (i == j) continue;

      glm::vec3 diff = _particles[j].position - _particles[i].position;

      // Handle wrapping
      if (wrapEdges) {
        if (diff.x > worldSize * 0.5f) diff.x -= worldSize;
        if (diff.x < -worldSize * 0.5f) diff.x += worldSize;
        if (diff.y > worldSize * 0.5f) diff.y -= worldSize;
        if (diff.y < -worldSize * 0.5f) diff.y += worldSize;
        if (diff.z > worldSize * 0.5f) diff.z -= worldSize;
        if (diff.z < -worldSize * 0.5f) diff.z += worldSize;
      }

      float distSq = glm::dot(diff, diff);
      if (distSq > radiusSq || distSq < 0.0001f) continue;

      float dist = std::sqrt(distSq);
      float normalizedDist = dist / interactionRadius;

      float attraction = _rules[_particles[i].colorIndex][_particles[j].colorIndex];
      float force = attractionForce(normalizedDist, attraction);

      totalForce += (diff / dist) * force;
    }

    _particles[i].velocity += totalForce * forceStrength * deltaTime * 60.0f;
  }
}

void ParticleLife3D::updatePositions(float deltaTime) {
  float halfWorld = worldSize * 0.5f;

  for (auto& p : _particles) {
    // Apply friction
    p.velocity *= (1.0f - friction);

    // Clamp velocity
    float speed = glm::length(p.velocity);
    if (speed > maxSpeed) {
      p.velocity = (p.velocity / speed) * maxSpeed;
    }

    // Update position
    p.position += p.velocity * deltaTime * 60.0f;

    // Wrap or clamp edges
    if (wrapEdges) {
      if (p.position.x > halfWorld) p.position.x -= worldSize;
      if (p.position.x < -halfWorld) p.position.x += worldSize;
      if (p.position.y > halfWorld) p.position.y -= worldSize;
      if (p.position.y < -halfWorld) p.position.y += worldSize;
      if (p.position.z > halfWorld) p.position.z -= worldSize;
      if (p.position.z < -halfWorld) p.position.z += worldSize;
    } else {
      p.position = glm::clamp(p.position, glm::vec3(-halfWorld),
                              glm::vec3(halfWorld));
      // Bounce off walls
      if (std::abs(p.position.x) >= halfWorld) p.velocity.x *= -0.5f;
      if (std::abs(p.position.y) >= halfWorld) p.velocity.y *= -0.5f;
      if (std::abs(p.position.z) >= halfWorld) p.velocity.z *= -0.5f;
    }
  }
}

glm::vec3 ParticleLife3D::getColor(int colorIndex) const {
  if (colorIndex < 0 || colorIndex >= NUM_COLORS) return glm::vec3(1.0f);
  return _colors[colorIndex];
}
