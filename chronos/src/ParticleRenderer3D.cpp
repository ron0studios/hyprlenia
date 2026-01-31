#include "ParticleRenderer3D.h"

#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "ParticleLifeCUDA.cuh"

namespace {
const char* vertexShaderSource = R"(
#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vColor;
out float vDepth;

uniform mat4 view;
uniform mat4 projection;
uniform float pointSize;

void main() {
    vec4 viewPos = view * vec4(aPos, 1.0);
    gl_Position = projection * viewPos;
    
    // Size attenuation based on distance
    float dist = length(viewPos.xyz);
    gl_PointSize = pointSize * (1.0 / (1.0 + dist * 0.5));
    
    vColor = aColor;
    vDepth = -viewPos.z;  // Depth for fog effect
}
)";

const char* fragmentShaderSource = R"(
#version 450 core
in vec3 vColor;
in float vDepth;

out vec4 FragColor;

uniform float glowIntensity;

void main() {
    // Create circular point with soft edges
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
        discard;
    }
    
    // Soft glow falloff
    float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
    float glow = exp(-dist * 3.0) * glowIntensity;
    
    // Core is brighter
    vec3 coreColor = vColor + vec3(0.3) * (1.0 - dist * 2.0);
    vec3 glowColor = vColor * glow;
    
    vec3 finalColor = coreColor + glowColor;
    
    // Slight depth fog for atmosphere
    float fog = exp(-vDepth * 0.1);
    finalColor = mix(vec3(0.02, 0.02, 0.05), finalColor, fog);
    
    FragColor = vec4(finalColor, alpha);
}
)";
}  // namespace

ParticleRenderer3D::ParticleRenderer3D() = default;

ParticleRenderer3D::~ParticleRenderer3D() { shutdown(); }

void ParticleRenderer3D::init() {
  // Compile shaders
  unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vertexShaderSource, nullptr);
  glCompileShader(vs);

  unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fragmentShaderSource, nullptr);
  glCompileShader(fs);

  _shaderProgram = glCreateProgram();
  glAttachShader(_shaderProgram, vs);
  glAttachShader(_shaderProgram, fs);
  glLinkProgram(_shaderProgram);

  glDeleteShader(vs);
  glDeleteShader(fs);

  // Get uniform locations
  _locView = glGetUniformLocation(_shaderProgram, "view");
  _locProjection = glGetUniformLocation(_shaderProgram, "projection");
  _locPointSize = glGetUniformLocation(_shaderProgram, "pointSize");
  _locGlowIntensity = glGetUniformLocation(_shaderProgram, "glowIntensity");

  // Create VAO/VBO
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);

  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);

  // Position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // Color attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
}

void ParticleRenderer3D::shutdown() {
  if (_shaderProgram) {
    glDeleteProgram(_shaderProgram);
    _shaderProgram = 0;
  }
  if (_vao) {
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;
  }
  if (_vbo) {
    glDeleteBuffers(1, &_vbo);
    _vbo = 0;
  }
}

void ParticleRenderer3D::renderCUDA(const std::vector<ParticleCUDA>& particles,
                                     glm::mat4 const& view,
                                     glm::mat4 const& projection, float pointSize,
                                     float glowIntensity) {
  if (particles.empty()) return;

  // Build vertex data
  std::vector<float> vertexData;
  vertexData.reserve(particles.size() * 6);

  for (auto const& p : particles) {
    glm::vec3 color = getParticleColor(p.colorIndex);
    vertexData.push_back(p.position.x);
    vertexData.push_back(p.position.y);
    vertexData.push_back(p.position.z);
    vertexData.push_back(color.r);
    vertexData.push_back(color.g);
    vertexData.push_back(color.b);
  }

  // Upload to GPU
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float),
               vertexData.data(), GL_DYNAMIC_DRAW);

  // Render
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // Additive blending for glow
  glDepthMask(GL_FALSE);  // Don't write to depth buffer

  glUseProgram(_shaderProgram);
  glUniformMatrix4fv(_locView, 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(_locProjection, 1, GL_FALSE, glm::value_ptr(projection));
  glUniform1f(_locPointSize, pointSize);
  glUniform1f(_locGlowIntensity, glowIntensity);

  glBindVertexArray(_vao);
  glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particles.size()));

  glDepthMask(GL_TRUE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBindVertexArray(0);
  glUseProgram(0);
}
