#include "ParticleSimulation.h"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>

namespace {
const char* particleVertexShader = R"(
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;

out vec3 vColor;

uniform vec2 resolution;

void main()
{
    vec2 ndc = (aPos / resolution) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = aSize;
    vColor = aColor;
}
)";

const char* particleFragmentShader = R"(
#version 450 core
in vec3 vColor;
out vec4 FragColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
        discard;
    }
    
    // Soft glow effect
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    float glow = exp(-dist * 4.0) * 0.5;
    
    vec3 color = vColor + vec3(glow);
    FragColor = vec4(color, alpha);
}
)";

const char* previewVertexShader = R"(
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* previewFragmentShader = R"(
#version 450 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;

void main()
{
    FragColor = texture(tex, TexCoord);
}
)";
}  // namespace

ParticleSimulation::ParticleSimulation(int width, int height)
    : _width(width), _height(height), _rng(std::random_device{}()) {
  initGL();
  initParticles();

  // Create some attractors for interesting emergent patterns
  _attractors.push_back({_width * 0.3f, _height * 0.3f, 50.0f, 0.0f});
  _attractors.push_back({_width * 0.7f, _height * 0.3f, 50.0f, 2.0f});
  _attractors.push_back({_width * 0.5f, _height * 0.7f, 50.0f, 4.0f});
  _attractors.push_back(
      {_width * 0.5f, _height * 0.5f, -30.0f, 1.0f});  // Repeller
}

ParticleSimulation::~ParticleSimulation() {
  if (_fbo) glDeleteFramebuffers(1, &_fbo);
  if (_outputTexture) glDeleteTextures(1, &_outputTexture);
  if (_shaderProgram) glDeleteProgram(_shaderProgram);
  if (_vao) glDeleteVertexArrays(1, &_vao);
  if (_vbo) glDeleteBuffers(1, &_vbo);
  if (_previewVao) glDeleteVertexArrays(1, &_previewVao);
  if (_previewVbo) glDeleteBuffers(1, &_previewVbo);
  if (_previewShader) glDeleteProgram(_previewShader);
}

void ParticleSimulation::initGL() {
  // Create output texture
  glGenTextures(1, &_outputTexture);
  glBindTexture(GL_TEXTURE_2D, _outputTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _width, _height, 0, GL_RGBA,
               GL_FLOAT, nullptr);

  // Create FBO
  glGenFramebuffers(1, &_fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         _outputTexture, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Compile particle shader
  unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &particleVertexShader, nullptr);
  glCompileShader(vs);

  unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &particleFragmentShader, nullptr);
  glCompileShader(fs);

  _shaderProgram = glCreateProgram();
  glAttachShader(_shaderProgram, vs);
  glAttachShader(_shaderProgram, fs);
  glLinkProgram(_shaderProgram);
  glDeleteShader(vs);
  glDeleteShader(fs);

  // Create particle VAO/VBO
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);

  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);

  // Position (x, y)
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // Color (r, g, b)
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // Size
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void*)(5 * sizeof(float)));
  glEnableVertexAttribArray(2);

  glBindVertexArray(0);

  // Compile preview shader
  vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &previewVertexShader, nullptr);
  glCompileShader(vs);

  fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &previewFragmentShader, nullptr);
  glCompileShader(fs);

  _previewShader = glCreateProgram();
  glAttachShader(_previewShader, vs);
  glAttachShader(_previewShader, fs);
  glLinkProgram(_previewShader);
  glDeleteShader(vs);
  glDeleteShader(fs);

  // Create preview quad
  float quadVertices[] = {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
                          1.0f,  1.0f,  1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f,
                          1.0f,  1.0f,  1.0f, 1.0f, -1.0f, 1.0f,  0.0f, 1.0f};

  glGenVertexArrays(1, &_previewVao);
  glGenBuffers(1, &_previewVbo);

  glBindVertexArray(_previewVao);
  glBindBuffer(GL_ARRAY_BUFFER, _previewVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
}

void ParticleSimulation::initParticles() {
  _particles.resize(_particleCount);

  std::uniform_real_distribution<float> xDist(0.0f, static_cast<float>(_width));
  std::uniform_real_distribution<float> yDist(0.0f,
                                              static_cast<float>(_height));
  std::uniform_real_distribution<float> velDist(-20.0f, 20.0f);
  std::uniform_real_distribution<float> hueDist(0.0f, 1.0f);
  std::uniform_real_distribution<float> sizeDist(2.0f, 6.0f);

  for (auto& p : _particles) {
    p.x = xDist(_rng);
    p.y = yDist(_rng);
    p.vx = velDist(_rng);
    p.vy = velDist(_rng);

    // Rainbow colors based on position
    float hue = hueDist(_rng);
    // HSV to RGB (simplified)
    float h = hue * 6.0f;
    float c = 1.0f;
    float x = c * (1.0f - std::abs(std::fmod(h, 2.0f) - 1.0f));
    if (h < 1) {
      p.r = c;
      p.g = x;
      p.b = 0;
    } else if (h < 2) {
      p.r = x;
      p.g = c;
      p.b = 0;
    } else if (h < 3) {
      p.r = 0;
      p.g = c;
      p.b = x;
    } else if (h < 4) {
      p.r = 0;
      p.g = x;
      p.b = c;
    } else if (h < 5) {
      p.r = x;
      p.g = 0;
      p.b = c;
    } else {
      p.r = c;
      p.g = 0;
      p.b = x;
    }

    p.life = 1.0f;
    p.size = sizeDist(_rng);
  }
}

void ParticleSimulation::update(float deltaTime) {
  _time += deltaTime;

  // Update attractor positions (orbiting)
  for (size_t i = 0; i < _attractors.size(); ++i) {
    auto& a = _attractors[i];
    float baseX = _width * 0.5f;
    float baseY = _height * 0.5f;
    float radius = _width * 0.25f;
    a.x = baseX + std::cos(_time * 0.5f + a.phase) * radius;
    a.y = baseY + std::sin(_time * 0.5f + a.phase) * radius;
  }

  // Update particles
  for (auto& p : _particles) {
    // Apply attractor forces
    for (auto const& a : _attractors) {
      float dx = a.x - p.x;
      float dy = a.y - p.y;
      float dist = std::sqrt(dx * dx + dy * dy) + 1.0f;
      float force = a.strength / (dist * dist) * 1000.0f;
      p.vx += (dx / dist) * force * deltaTime;
      p.vy += (dy / dist) * force * deltaTime;
    }

    // Apply some friction
    p.vx *= 0.99f;
    p.vy *= 0.99f;

    // Clamp velocity
    float speed = std::sqrt(p.vx * p.vx + p.vy * p.vy);
    if (speed > 200.0f) {
      p.vx = (p.vx / speed) * 200.0f;
      p.vy = (p.vy / speed) * 200.0f;
    }

    // Update position
    p.x += p.vx * deltaTime;
    p.y += p.vy * deltaTime;

    // Wrap around edges
    if (p.x < 0) p.x += _width;
    if (p.x >= _width) p.x -= _width;
    if (p.y < 0) p.y += _height;
    if (p.y >= _height) p.y -= _height;

    // Update color based on velocity (shifting hue)
    float hue = std::fmod(
        std::atan2(p.vy, p.vx) / (2.0f * 3.14159f) + 0.5f + _time * 0.1f, 1.0f);
    float h = hue * 6.0f;
    float c = 0.8f + 0.2f * (speed / 200.0f);
    float x = c * (1.0f - std::abs(std::fmod(h, 2.0f) - 1.0f));
    if (h < 1) {
      p.r = c;
      p.g = x;
      p.b = 0;
    } else if (h < 2) {
      p.r = x;
      p.g = c;
      p.b = 0;
    } else if (h < 3) {
      p.r = 0;
      p.g = c;
      p.b = x;
    } else if (h < 4) {
      p.r = 0;
      p.g = x;
      p.b = c;
    } else if (h < 5) {
      p.r = x;
      p.g = 0;
      p.b = c;
    } else {
      p.r = c;
      p.g = 0;
      p.b = x;
    }
  }

  renderToTexture();
}

void ParticleSimulation::renderToTexture() {
  // Upload particle data to GPU
  std::vector<float> vertexData;
  vertexData.reserve(_particles.size() * 6);
  for (auto const& p : _particles) {
    vertexData.push_back(p.x);
    vertexData.push_back(p.y);
    vertexData.push_back(p.r);
    vertexData.push_back(p.g);
    vertexData.push_back(p.b);
    vertexData.push_back(p.size);
  }

  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float),
               vertexData.data(), GL_DYNAMIC_DRAW);

  // Render to FBO
  glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
  glViewport(0, 0, _width, _height);
  glClearColor(0.0f, 0.0f, 0.02f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // Additive blending for glow
  glEnable(GL_PROGRAM_POINT_SIZE);

  glUseProgram(_shaderProgram);
  glUniform2f(glGetUniformLocation(_shaderProgram, "resolution"),
              static_cast<float>(_width), static_cast<float>(_height));

  glBindVertexArray(_vao);
  glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(_particles.size()));

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void ParticleSimulation::renderPreview() {
  glUseProgram(_previewShader);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, _outputTexture);
  glUniform1i(glGetUniformLocation(_previewShader, "tex"), 0);

  glBindVertexArray(_previewVao);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
}

void ParticleSimulation::setParticleCount(int count) {
  _particleCount = count;
  initParticles();
}
